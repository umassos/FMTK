from transformers import AutoProcessor, AutoModelForCausalLM
import os
import time
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class PhiModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../..', 'FMaaS-motivation/vqa/updated/models')
        if model_name=="phi":
            model_id='microsoft/Phi-3.5-vision-instruct'
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": self.device})

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
                use_cache=False,
            )

            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            responses.append(response)
        return responses

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers


    def predict(self, dataloader, logger=None):
        """
        Run inference over a DataLoader, optionally logging per-sample
        VLM metrics (latency, tokens, GPU utilisation) via the FMTK Logger.
        """
        predictions = []
        labels = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            image, question, gt = batch['x'], batch['question'], batch['y']

            gpu_mem_before = logger.get_gpu_mem_mb() if logger else 0
            t0 = time.time()

            with torch.no_grad():
                answer = self.forward((image, question))

            latency_ms = (time.time() - t0) * 1000

            if logger:
                logger.log_vlm_sample(
                    latency_ms=latency_ms,
                    prompt_tokens=len(question[0].split()),
                    gen_tokens=len(answer[0].split()),
                    gpu_util_pct=logger.get_gpu_util_pct(),
                    gpu_mem_delta_mb=logger.get_gpu_mem_mb() - gpu_mem_before,
                )

            predictions.append(answer)
            labels.append(gt)
        return predictions, labels
