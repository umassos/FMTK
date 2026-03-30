from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import numpy as np
from fmtk.components.base import BaseModel
from tqdm import tqdm

PHI3_MODELS = {
    'phi3-mini':   'microsoft/Phi-3-mini-4k-instruct',
    'phi3-small':  'microsoft/Phi-3-small-8k-instruct',
    'phi3-medium': 'microsoft/Phi-3-medium-4k-instruct',
}

class Phi3Model(BaseModel):
    def __init__(self, device, model_name='phi3-mini', model_config=None):
        super().__init__()
        self.device = device
        self.model_category = 'llm'
        model_config = model_config or {}
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.abspath(os.path.join(base_dir, '../../../../models/llm/pretrained'))

        model_id = PHI3_MODELS.get(model_name, model_name)
        print(f"[Phi-3] Loading {model_id} on device {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=models_directory, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=models_directory,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        self.model.eval()

        self.max_new_tokens = model_config.get('max_new_tokens', 64)

    def preprocess(self, batch_x, mask=None):
        return batch_x, mask

    def forward(self, batch_x, mask=None):
        """
        batch_x: list of prompt strings.
        Returns list of raw generated strings.
        """
        (_,batch_x_prompt)=batch_x
        responses = []
        for prompt in batch_x_prompt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = outputs[0][input_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            responses.append(response)
        return responses

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
