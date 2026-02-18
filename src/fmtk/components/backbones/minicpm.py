from transformers import AutoModel, AutoTokenizer
import os
import time
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class MinicpmModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../..', 'FMaaS-motivation/vqa/updated/models')
        if model_name=="minicpm":
            model_id='openbmb/MiniCPM-V-2_6'
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=models_directory)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=models_directory).eval().to(self.device)
        self.processor = tokenizer

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            msgs = [{"role": "user", "content": [image, question]}]
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.processor,
            )
            responses.append(response)
        return responses

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.strip().split()[0]
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
