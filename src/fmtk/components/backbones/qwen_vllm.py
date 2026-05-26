import os
import uuid
import time
import numpy as np
from tqdm import tqdm
from fmtk.components.base import BaseModel
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from transformers import AutoTokenizer

QWEN_TEXT_MODELS = {
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b':   'Qwen/Qwen2.5-3B-Instruct',
    'qwen2.5-7b':   'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b':  'Qwen/Qwen2.5-14B-Instruct',
}

class QwenVLLMModel(BaseModel):
    """
    Qwen2.5 backbone backed by vLLM for faster inference.

    Uses vllm.LLM (synchronous batch API) which applies PagedAttention
    and continuous batching internally — the full batch from each DataLoader
    step is scheduled together, giving significantly higher throughput than
    the per-sample HF generate() loop in QwenTextModel.

    All Qwen2.5 variants are supported (vLLM >= 0.6.6 + torch 2.5+cu121):
        qwen2.5-0.5b, 1.5b, 3b, 7b, 14b

    Differences from QwenTextModel (HF):
    - Uses vllm.LLM for batched generation instead of per-sample HF generate()
    - No PyTorch tensors are passed around; inputs/outputs are strings
    - Not suitable for gradient-based decoder training (embeddings not exposed)
    - device should be a CUDA device string, e.g. 'cuda:0' or 'cuda'

    Usage:
        model = QwenVLLMModel(device='cuda:0', model_name='qwen2.5-7b',
                              model_config={'max_new_tokens': 128})
        pipeline = Pipeline(model)
        labels, preds = pipeline.predict(test_loader, cfg={})
    """

    def __init__(self, device, model_name='qwen2.5-7b', model_config=None, async_only=False):
        super().__init__()
        self.device = device
        # vLLM 0.6.x selects the GPU via CUDA_VISIBLE_DEVICES; the 'device'
        # kwarg was added in later versions and is not available here.
        # Only set CUDA_VISIBLE_DEVICES if not already set externally (e.g. by
        # the SSH launcher), so the caller controls which physical GPU is used.
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            gpu_index = int(device.split(':')[1]) if ':' in device else 0
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

        self.model_category = 'llm'
        model_config = model_config or {}
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../models/llm/pretrained')
        model_id = QWEN_TEXT_MODELS.get(model_name, model_name)
        print(f"[Qwen vLLM] Loading {model_id} on device {device}")
        # Load tokenizer once for use in preprocess (needed by both sync and async paths)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.sampling_params = SamplingParams(
            temperature=model_config.get('temperature', 0.0),
            max_tokens=model_config.get('max_new_tokens', 64),
        )
        if async_only:
            # AsyncLLMEngine is created lazily in async_forward() to avoid
            # loading two copies of the model into GPU memory simultaneously.
            self._async_engine = None
            self._async_engine_args = dict(
                model=model_id,
                download_dir=models_directory,
                dtype='float16',
                trust_remote_code=True,
                gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.85),
                tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                enforce_eager=model_config.get('enforce_eager', True),
            )
            if 'max_model_len' in model_config:
                self._async_engine_args['max_model_len'] = model_config['max_model_len']

            # Multi-LoRA: enable on the engine when requested. Per-request
            # adapter routing is done via async_forward(prompt, adapter_name).
            if bool(model_config.get('enable_lora', False)):
                self._async_engine_args['enable_lora'] = True
                self._async_engine_args['max_loras'] = int(model_config.get('max_loras', 4))
                self._async_engine_args['max_lora_rank'] = int(model_config.get('max_lora_rank', 64))
                if 'max_cpu_loras' in model_config:
                    self._async_engine_args['max_cpu_loras'] = int(model_config['max_cpu_loras'])

            self._async_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**self._async_engine_args)
            )

        else:
            self.llm = LLM(
                model=model_id,
                download_dir=models_directory,
                dtype='float16',
                trust_remote_code=True,
                gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.85),
                tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                enforce_eager=model_config.get('enforce_eager', True),
            )

        # Multi-LoRA bookkeeping (used by both async and sync paths if enabled).
        self._loras: dict[str, LoRARequest] = {}
        self._next_lora_id = 1

    def preprocess(self, batch_x, mask=None):
        """Apply the Qwen chat template to each prompt string."""
        formatted = []
        for prompt in batch_x:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(text)
        return formatted, mask

    def forward(self, batch_x, mask=None):
        """
        batch_x: list of raw prompt strings (chat template applied inside).
        Returns list of generated strings (no prompt prefix).
        """
        formatted, _ = self.preprocess(batch_x, mask)
        outputs = self.llm.generate(formatted, self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def register_adapter(self, adapter_name: str, lora_path: str) -> str:
        """Register a LoRA adapter directory (PEFT format) under `adapter_name`.
        Engine must have been built with enable_lora=True. Returns the name."""
        if adapter_name in self._loras:
            return adapter_name
        lora_int_id = self._next_lora_id
        self._next_lora_id += 1
        self._loras[adapter_name] = LoRARequest(adapter_name, lora_int_id, lora_path)
        print(f"[Qwen vLLM] Registered LoRA '{adapter_name}' id={lora_int_id} path={lora_path}")
        return adapter_name

    def unregister_adapter(self, adapter_name: str) -> None:
        self._loras.pop(adapter_name, None)

    async def async_forward(self, prompt: str, adapter_name: str | None = None) -> str:
        """
        Schedule a single prompt into the AsyncLLMEngine and await its result.
        If `adapter_name` is given and registered, the request is routed through
        that LoRA. Concurrent calls are batched by vLLM (continuous batching).
        """

        formatted, _ = self.preprocess([prompt])
        request_id = str(uuid.uuid4())
        lora_request = self._loras.get(adapter_name) if adapter_name else None

        final_output = None
        async for output in self._async_engine.generate(
            formatted[0], self.sampling_params, request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = output

        text = final_output.outputs[0].text.strip() if final_output else ""
        return self.postprocess([text])[0]

    def postprocess(self, responses):
        return responses

    def predict(self, dataloader):
        all_preds, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            prompts = batch['x']
            labels = batch['y']
            responses = self.forward(prompts)
            responses = self.postprocess(responses)
            all_preds.extend(responses)
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(list(labels))
        return np.array(all_labels, dtype=object), np.array(all_preds, dtype=object)
