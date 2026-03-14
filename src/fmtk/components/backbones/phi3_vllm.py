import os
import uuid
import time
import numpy as np
from tqdm import tqdm
from fmtk.components.base import BaseModel
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import torch
from transformers import AutoTokenizer

PHI3_MODELS = {
    'phi3-mini':   'microsoft/Phi-3-mini-4k-instruct',
    'phi3-small':  'microsoft/Phi-3-small-8k-instruct',
    'phi3-medium': 'microsoft/Phi-3-medium-4k-instruct',
}

class Phi3VLLMModel(BaseModel):
    """
    Phi-3 backbone backed by vLLM for faster inference.

    Uses vllm.LLM (synchronous batch API) which applies PagedAttention
    and continuous batching internally — the full batch from each DataLoader
    step is scheduled together, giving significantly higher throughput than
    the per-sample HF generate() loop in Phi3Model.

    All three Phi-3 variants are supported (vLLM >= 0.6.6 + torch 2.5+cu121):
        phi3-mini   : sliding window 2047 — supported by FlashAttention-2
        phi3-small  : sliding window — supported
        phi3-medium : no sliding window — supported

    Differences from Phi3Model (HF):
    - Uses vllm.LLM for batched generation instead of per-sample HF generate()
    - No PyTorch tensors are passed around; inputs/outputs are strings
    - Not suitable for gradient-based decoder training (embeddings not exposed)
    - device should be a CUDA device string, e.g. 'cuda:0' or 'cuda'

    Usage:
        model = Phi3VLLMModel(device='cuda:0', model_name='phi3-mini',
                              model_config={'max_new_tokens': 128})
        pipeline = Pipeline(model)
        labels, preds = pipeline.predict(test_loader, cfg={})
    """

    def __init__(self, device, model_name='phi3-mini', model_config=None):
        super().__init__()
        self.device = device
        # vLLM 0.6.x selects the GPU via CUDA_VISIBLE_DEVICES; the 'device'
        # kwarg was added in later versions and is not available here.
        gpu_index = int(device.split(':')[1]) if ':' in device else 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

        self.model_category = 'llm'
        model_config = model_config or {}
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../models/llm/pretrained')
        model_id = PHI3_MODELS.get(model_name, model_name)
        print(f"[Phi-3 vLLM] Loading {model_id} on device {device}")
        # Load tokenizer once for use in preprocess (needed by both sync and async paths)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.sampling_params = SamplingParams(
            temperature=model_config.get('temperature', 0.0),
            max_tokens=model_config.get('max_new_tokens', 64),
        )
        if model_config.get('type')=='async':
            # AsyncLLMEngine is created lazily in async_forward() to avoid
            # loading two copies of the model into GPU memory simultaneously.
            self._async_engine = None
            self._async_engine_args = dict(
                model=model_id,
                download_dir=models_directory,
                dtype='bfloat16',
                trust_remote_code=True,
                gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.85),
                tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                enforce_eager=model_config.get('enforce_eager', True),
            )
            self._async_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**self._async_engine_args)
            )
        else:
            self.llm = LLM(
                model=model_id,
                download_dir=models_directory,
                dtype='bfloat16',
                trust_remote_code=True,
                gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.85),
                tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                enforce_eager=model_config.get('enforce_eager', True),
            )

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def preprocess(self, batch_x, mask=None):
        """Apply the Phi-3 chat template to each prompt string."""
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

    async def async_forward(self, prompt: str) -> dict:
        """
        Schedule a single prompt into the AsyncLLMEngine and await its result.

        Multiple concurrent calls to async_forward() are batched at the
        iteration level by vLLM (true continuous batching). Returns a dict
        with text_output and timing fields compatible with InferResponse.
        """

        formatted, _ = self.preprocess([prompt])
        request_id = str(uuid.uuid4())

        final_output = None
        async for output in self._async_engine.generate(
            formatted[0], self.sampling_params, request_id=request_id
        ):
            final_output = output

        return self.postprocess(final_output)
    
    def postprocess(self, responses):
        text = [responses.outputs[0].text.strip() if responses else ""] 
        return text[0]

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
