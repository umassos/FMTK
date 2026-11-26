"""Qwen2-VL backbone on vLLM with per-request multi-LoRA routing.

This is the union of two existing backbones in this package:

  - ``phi.PhiVLLMModel``  contributes the multimodal path: images become
    ``{'prompt': ..., 'multi_modal_data': {'image': PIL}}`` inputs.
  - ``qwen_vllm.QwenVLLMModel`` contributes multi-LoRA: a ``_loras`` registry
    plus ``register_adapter``, so ``async_forward(..., adapter_name=...)``
    routes each request through its own ``LoRARequest``. ``PhiVLLMModel`` only
    has a single ``self.lora_request`` slot and cannot serve N tasks.

``model_category`` is 'vlm', which is what makes ``Pipeline.add_adapter``
resolve adapter directories under ``models/vlm/finetuned/<path>``.

LoRA scope
----------
vLLM 0.6.6 lists exactly {qkv_proj, o_proj, gate_up_proj, down_proj} in
``Qwen2VLForConditionalGeneration.supported_lora_modules``; the vision encoder
is an explicit upstream TODO. Adapters must therefore be language-side only.
Adapters built by ``experiments/RTVSntask/vlm/make_loras.py`` target
q_proj/v_proj, which in Qwen2-VL occur only under ``model.layers.*.self_attn``
(the visual tower uses ``visual.blocks.*.attn.qkv``), so they satisfy this.

The min_pixels/max_pixels cap is REQUIRED, not tuning
-----------------------------------------------------
With ``enable_lora=True`` and default pixel limits, engine construction dies
during ``profile_run`` with:

    RuntimeError: The size of tensor a (5120) must match the size of
                  tensor b (16384) at non-singleton dimension 0

That is not an adapter shape problem despite how it reads (and despite
RTVSntask/run.sh's ``classify_failure`` bucketing it as
``adapter_shape_mismatch``). vLLM builds a dummy profiling image at Qwen2-VL's
default ``max_pixels``, reserving 16384 vision tokens against a 5120-token
batch budget. Capping ``max_pixels`` shrinks that dummy image and the build
succeeds. Verified: no-LoRA builds, LoRA+default-pixels fails,
LoRA+capped-pixels builds clean.
"""
import os
import uuid
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from fmtk.components.base import BaseModel

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

# Qwen2-VL only. Qwen2.5-VL is NOT usable on this stack: vLLM 0.6.6 has no
# qwen2_5_vl module and does not register Qwen2_5_VLForConditionalGeneration,
# and transformers 4.48 lacks the class. Both would need upgrading, which
# breaks the TPC hooks in device/runtime.py that patch vLLM 0.6.6 internals
# by name. Listing a 2.5 id here would fail confusingly at engine build.
QWEN_VL_MODELS = {
    'qwen2-vl-2b': 'Qwen/Qwen2-VL-2B-Instruct',
    'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
}

# 28x28 MNIST OCR digits: smart_resize floors them at 56x56 -> 4 merged tokens.
# The ceiling exists to bound the profile_run dummy image (see module docstring).
_DEFAULT_MIN_PIXELS = 56 * 56
_DEFAULT_MAX_PIXELS = 256 * 28 * 28


class Qwen2VLVLLMModel(BaseModel):
    """Qwen2-VL served by vLLM's AsyncLLMEngine, with multi-LoRA.

    Usage:
        model = Qwen2VLVLLMModel(
            device='cuda:0', model_name='qwen2-vl-2b',
            model_config={'enable_lora': True, 'max_loras': 8,
                          'max_lora_rank': 16, 'max_model_len': 512},
        )
        pipeline = Pipeline(model)
        pipeline.add_adapter(peft_cfg, load=True, train=False, path='qwenvlA')
        text = await model.async_forward((image, question), adapter_name='adapter_0')
    """

    def __init__(self, device, model_name='qwen2-vl-2b', model_config=None,
                 async_only=True):
        from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from transformers import AutoProcessor

        super().__init__()
        self.device = device
        self.model_category = 'vlm'
        model_config = model_config or {}

        # vLLM 0.6.x selects the GPU via CUDA_VISIBLE_DEVICES; the 'device'
        # kwarg arrived in later versions. Only set it if the caller has not
        # already (device/runtime.py VLLMRuntime.load does this itself).
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            gpu_index = int(device.split(':')[1]) if ':' in device else 0
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

        model_id = QWEN_VL_MODELS.get(model_name, model_name)
        print(f"[Qwen2-VL vLLM] Loading {model_id} on device {device}")

        # The processor (not just the tokenizer) is needed: apply_chat_template
        # has to emit the <|vision_start|><|image_pad|><|vision_end|> block.
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=_MODEL_CACHE, trust_remote_code=True,
        )

        # ignore_eos forces every request to emit exactly max_tokens.
        # The OCR prompt ("answer in one word") otherwise stops at ~2 tokens,
        # making a VLM request ~16x cheaper in decode than the LLM arm's
        # 32-token requests — a confound when comparing the two modalities.
        # Forcing the length makes output length an experimental constant.
        # Generated text past the answer is filler; this mode is for timing,
        # not accuracy (the adapters are random-init anyway).
        self.sampling_params = SamplingParams(
            temperature=model_config.get('temperature', 0.0),
            max_tokens=model_config.get('max_new_tokens', 64),
            ignore_eos=bool(model_config.get('ignore_eos', False)),
        )

        engine_kwargs = dict(
            model=model_id,
            download_dir=_MODEL_CACHE,
            dtype=model_config.get('dtype', 'float16'),
            trust_remote_code=True,
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.85),
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            enforce_eager=model_config.get('enforce_eager', False),
            limit_mm_per_prompt={'image': model_config.get('max_images_per_prompt', 1)},
            mm_processor_kwargs={
                'min_pixels': model_config.get('min_pixels', _DEFAULT_MIN_PIXELS),
                'max_pixels': model_config.get('max_pixels', _DEFAULT_MAX_PIXELS),
            },
        )
        if 'max_model_len' in model_config:
            engine_kwargs['max_model_len'] = model_config['max_model_len']

        # Multi-LoRA: per-request routing happens in async_forward via
        # LoRARequest; nothing is merged into the base weights.
        if bool(model_config.get('enable_lora', False)):
            engine_kwargs['enable_lora'] = True
            engine_kwargs['max_loras'] = int(model_config.get('max_loras', 4))
            engine_kwargs['max_lora_rank'] = int(model_config.get('max_lora_rank', 16))
            if 'max_cpu_loras' in model_config:
                engine_kwargs['max_cpu_loras'] = int(model_config['max_cpu_loras'])

        if async_only:
            self._async_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**engine_kwargs)
            )
            self.llm = None
        else:
            self._async_engine = None
            self.llm = LLM(**engine_kwargs)

        self._loras: dict = {}
        self._next_lora_id = 1

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def _to_pil(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(image, torch.Tensor):
            # Drop a leading batch dim if the caller passed (1, C, H, W).
            if image.dim() == 4 and image.shape[0] == 1:
                image = image[0]
            image = transforms.ToPILImage()(image)
        return image.convert('RGB')

    def preprocess(self, batch_x, mask=None):
        """batch_x is an (images, questions) pair.
        Returns (prompts, multi_modal_data_list, mask)."""
        images, questions = batch_x
        prompts, mm_data = [], []
        for image, question in zip(images, questions):
            image = self._to_pil(image)
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(text)
            mm_data.append({'image': image})
        return prompts, mm_data, mask

    # ------------------------------------------------------------------
    # Adapters
    # ------------------------------------------------------------------

    def register_adapter(self, adapter_name: str, lora_path: str) -> str:
        """Register a PEFT adapter directory under `adapter_name`. The engine
        must have been built with enable_lora=True. Returns the name.

        Presence of this method is what makes Pipeline.add_adapter take the
        vLLM branch instead of wrapping the weights with HF PEFT.
        """
        from vllm.lora.request import LoRARequest
        if adapter_name in self._loras:
            return adapter_name
        lora_int_id = self._next_lora_id
        self._next_lora_id += 1
        self._loras[adapter_name] = LoRARequest(adapter_name, lora_int_id, lora_path)
        print(f"[Qwen2-VL vLLM] Registered LoRA '{adapter_name}' "
              f"id={lora_int_id} path={lora_path}")
        return adapter_name

    def unregister_adapter(self, adapter_name: str) -> None:
        self._loras.pop(adapter_name, None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_x, mask=None):
        """Synchronous batched generate. batch_x is an (images, questions) pair.
        Only available when constructed with async_only=False."""
        if self.llm is None:
            raise RuntimeError("forward() requires async_only=False; "
                               "use async_forward() with the async engine.")
        prompts, mm_data, _ = self.preprocess(batch_x, mask)
        inputs = [{'prompt': p, 'multi_modal_data': m}
                  for p, m in zip(prompts, mm_data)]
        outputs = self.llm.generate(inputs, self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    async def async_forward(self, batch_x, adapter_name: str | None = None) -> str:
        """Schedule one (image, question) pair into the AsyncLLMEngine and
        await its result. Concurrent calls are batched at the iteration level
        by vLLM — continuous batching, mixed adapters in one batch."""
        image, question = batch_x
        prompts, mm_data, _ = self.preprocess(([image], [question]))
        inputs = {'prompt': prompts[0], 'multi_modal_data': mm_data[0]}

        lora_request = self._loras.get(adapter_name) if adapter_name else None
        request_id = str(uuid.uuid4())

        final_output = None
        async for output in self._async_engine.generate(
            inputs, self.sampling_params, request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = output

        return final_output.outputs[0].text.strip() if final_output else ""

    # ------------------------------------------------------------------
    # Postprocess / predict
    # ------------------------------------------------------------------

    def postprocess(self, responses):
        if isinstance(responses, str):
            return responses.split()[0] if responses else ""
        return [r.split()[0] if r else "" for r in responses]

    def enable_peft(self, _):
        raise NotImplementedError(
            "Qwen2VLVLLMModel does not support PEFT training. Train with "
            "QwenModel (HF) and save the adapter, or generate one with "
            "experiments/RTVSntask/vlm/make_loras.py, then register it via "
            "Pipeline.add_adapter(train=False)."
        )

    def predict(self, dataloader):
        all_preds, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            responses = self.forward((batch['x'], batch['question']))
            all_preds.extend(self.postprocess(responses))
            labels = batch['y']
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(list(labels))
        return np.array(all_labels, dtype=object), np.array(all_preds, dtype=object)
