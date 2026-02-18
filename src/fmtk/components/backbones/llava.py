from transformers import LlavaForConditionalGeneration, AutoProcessor, LlavaNextForConditionalGeneration
import os
import time
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class LlavaModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../..', 'FMaaS-motivation/vqa/updated/models')
        if model_name=="llava-1.5-7b":
            model_id='llava-hf/llava-1.5-7b-hf'
        elif model_name=="llava-1.5-13b":
            model_id='llava-hf/llava-1.5-13b-hf'
        elif model_name=="llava-v1.6-13b":
            model_id='llava-hf/llava-v1.6-vicuna-13b-hf'

        if model_name in ("llava-1.5-7b", "llava-1.5-13b"):
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory)
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, device_map={"": self.device})
        elif model_name=="llava-v1.6-13b":
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True, device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=20)
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            responses.append(response)
        return responses

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split("ASSISTANT:")[-1].strip().split()[0]
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
