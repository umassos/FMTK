from transformers import AutoProcessor, AutoModelForCausalLM
import os
import uuid
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import re
from fmtk.components.base import BaseModel
from peft import get_peft_model

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

PHI_MODELS = {
    'phi3.5-vision': 'microsoft/Phi-3.5-vision-instruct',
    'phi3-mini':     'microsoft/Phi-3-mini-4k-instruct',
    'phi3-small':    'microsoft/Phi-3-small-8k-instruct',
    'phi3-medium':   'microsoft/Phi-3-medium-4k-instruct',
}
_PHI_VLM_MODELS = {'phi3.5-vision'}

class PhiModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlm'
        models_directory = _MODEL_CACHE
        if model_name in ("phi-3.5-vision-instruct"):
            model_id='microsoft/Phi-3.5-vision-instruct'
        self.peft_enable = False
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            prompt = f"<|image_1|>\n{question}"
            messages = [{"role": "user", "content": prompt}]

            text_prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text_prompt, [image], return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=20,
                do_sample=False,
            )

            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            responses.append(response)
        return responses

    def train_step(self, batch):
        """Single training step — returns scalar loss via causal LM objective."""
        image, question, label = batch['x'], batch['question'], batch['y']
        losses = []
        for img, q, lbl in zip(image, question, label):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            prompt = f"<|image_1|>\n{q}"
            messages = [{"role": "user", "content": prompt}]
            text_prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = text_prompt + lbl + self.processor.tokenizer.eos_token
            inputs = self.processor(full_text, [img], return_tensors="pt").to(self.device)
            labels = inputs["input_ids"].clone()
            # mask the prompt tokens so loss is only on the answer
            prompt_ids = self.processor(text_prompt, [img], return_tensors="pt")["input_ids"]
            labels[:, :prompt_ids.shape[1]] = -100
            outputs = self.model(**inputs, labels=labels)
            losses.append(outputs.loss)
        return torch.stack(losses).mean()

    def enable_peft(self, peft_cfg):
        # dispatch hooks from device_map interfere with get_peft_model — remove them first
        self.model = self.model.to(self.device)
        self.model = get_peft_model(self.model, peft_cfg)
        # LoRA params must be fp32 for GradScaler to unscale correctly
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        self.peft_enable = True

    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)

    def load_adapter(self, adapter_dir, peft_cfg=None):
        """Load a saved PEFT adapter for inference."""
        if not self.peft_enable:
            self.enable_peft(peft_cfg)
        self.model.load_adapter(adapter_dir, adapter_name='loaded')

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers


# ---------------------------------------------------------------------------
# vLLM runtime
# ---------------------------------------------------------------------------

class PhiVLLMModel(BaseModel):
    """
    Phi backbone backed by vLLM for faster inference.

    Supports all Phi variants via PHI_MODELS registry:
        phi3.5-vision : Phi-3.5-vision-instruct  (vlm — image + text input)
        phi3-mini     : Phi-3-mini-4k-instruct    (llm — text only)
        phi3-small    : Phi-3-small-8k-instruct   (llm — text only)
        phi3-medium   : Phi-3-medium-4k-instruct  (llm — text only)

    model_category is inferred automatically from model_name.

    Usage:
        # VLM
        model = PhiVLLMModel(device='cuda:0', model_name='phi3.5-vision',
                             model_config={'max_new_tokens': 64})
        # LLM
        model = PhiVLLMModel(device='cuda:0', model_name='phi3-mini',
                             model_config={'max_new_tokens': 128})
        pipeline = Pipeline(model)
        labels, preds = pipeline.predict(test_loader, cfg={})
    """

    def __init__(self, device, model_name='phi3.5-vision', model_config=None):
        from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from transformers import AutoTokenizer

        super().__init__()
        self.device = device
        model_config = model_config or {}

        model_id = PHI_MODELS.get(model_name, model_name)
        self.model_category = 'vlm' if model_name in _PHI_VLM_MODELS else 'llm'
        models_directory = _MODEL_CACHE

        # vLLM 0.6.x selects GPU via CUDA_VISIBLE_DEVICES
        gpu_index = int(device.split(':')[1]) if ':' in device else 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)

        print(f"[Phi vLLM] Loading {model_id} ({self.model_category}) on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sampling_params = SamplingParams(
            temperature=model_config.get('temperature', 0.0),
            max_tokens=model_config.get('max_new_tokens', 64),
        )

        self.lora_request = None

        engine_kwargs = dict(
            model=model_id,
            download_dir=models_directory,
            dtype=model_config.get('dtype', 'half'),
            trust_remote_code=True,
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.85),
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            enforce_eager=model_config.get('enforce_eager', True),
            enable_lora=model_config.get('enable_lora', False),
            **({'max_model_len': model_config['max_model_len']} if 'max_model_len' in model_config else {}),
        )

        if model_config.get('type') == 'async':
            self._async_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**engine_kwargs)
            )
            self.llm = None
        else:
            self.llm = LLM(**engine_kwargs)
            self._async_engine = None

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, batch_x, mask=None):
        """
        For LLM: batch_x is a list of prompt strings.
        For VLM: batch_x is a (images, questions) tuple.
        Returns (prompts, multi_modal_data_list, mask).
        """
        if self.model_category == 'vlm':
            images, questions = batch_x
            prompts, mm_data = [], []
            for image, question in zip(images, questions):
                if isinstance(image, torch.Tensor):
                    image = transforms.ToPILImage()(image)
                prompt = f"<|image_1|>\n{question}"
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(text)
                mm_data.append({'image': image})
            return prompts, mm_data, mask
        else:
            prompts = []
            for prompt in batch_x:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(text)
            return prompts, None, mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_x, mask=None):
        if self.model_category == 'vlm':
            return self._forward_vlm(batch_x, mask)
        return self._forward_llm(batch_x, mask)

    def _forward_llm(self, batch_x, mask=None):
        prompts, _, _ = self.preprocess(batch_x, mask)
        outputs = self.llm.generate(prompts, self.sampling_params,
                                    lora_request=self.lora_request)
        return [out.outputs[0].text.strip() for out in outputs]

    def _forward_vlm(self, batch_x, mask=None):
        prompts, mm_data, _ = self.preprocess(batch_x, mask)
        inputs = [
            {'prompt': p, 'multi_modal_data': m}
            for p, m in zip(prompts, mm_data)
        ]
        outputs = self.llm.generate(inputs, self.sampling_params,
                                    lora_request=self.lora_request)
        return [out.outputs[0].text.strip() for out in outputs]

    # ------------------------------------------------------------------
    # Async forward
    # ------------------------------------------------------------------

    async def async_forward(self, batch_x):
        """Single-sample async forward for use with AsyncLLMEngine."""
        if self.model_category == 'vlm':
            prompts, mm_data, _ = self.preprocess(
                ([batch_x[0]], [batch_x[1]])
            )
            inputs = {'prompt': prompts[0], 'multi_modal_data': mm_data[0]}
        else:
            prompts, _, _ = self.preprocess([batch_x])
            inputs = prompts[0]

        request_id = str(uuid.uuid4())
        final_output = None
        async for output in self._async_engine.generate(
            inputs, self.sampling_params, request_id=request_id,
            lora_request=self.lora_request,
        ):
            final_output = output

        return final_output.outputs[0].text.strip() if final_output else ""

    # ------------------------------------------------------------------
    # Postprocess / predict
    # ------------------------------------------------------------------

    def postprocess(self, responses):
        return [r.split()[0] if r else "" for r in responses]

    def load_adapter(self, adapter_dir, peft_cfg=None):
        """Load a pre-trained LoRA adapter saved by PhiModel for vLLM inference.

        The engine must have been initialised with enable_lora=True
        (set model_config={'enable_lora': True} in config.py).
        """
        from vllm.lora.request import LoRARequest
        self.lora_request = LoRARequest('adapter', 1, adapter_dir)

    def enable_peft(self, _):
        raise NotImplementedError(
            "PhiVLLMModel does not support PEFT training. "
            "Train with PhiModel (HF) and save the adapter, then load it "
            "via Pipeline.add_adapter() with train=False."
        )

    def predict(self, dataloader):
        all_preds, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            if self.model_category == 'vlm':
                batch_x = (batch['x'], batch['question'])
            else:
                batch_x = batch['x']
            labels = batch['y']
            responses = self.forward(batch_x)
            responses = self.postprocess(responses)
            all_preds.extend(responses)
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(list(labels))
        return np.array(all_labels, dtype=object), np.array(all_preds, dtype=object)
