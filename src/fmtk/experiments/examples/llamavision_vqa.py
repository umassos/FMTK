from timeseries.metrics import get_mae, get_accuracy
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.vqa import VQADataset
from timeseries.components.backbones.llama_vision import LlamaVisionModel
from huggingface_hub import login

with open("../../hf-token.txt", "r") as f:
    hf_token = f.read().strip()
login(token=hf_token)

device='cuda:0'

task_cfg={
    'inference_config': {
        'batch_size': 1,
        'shuffle':False
        },    
    }  
dataset_cfg={
        'dataset_path': '../../../dataset/val2014',
      
} 

dataloader_test = DataLoader(VQADataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

P=Pipeline(LlamaVisionModel(device,'llama-vision'))
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_accuracy(y_test, y_pred)
print(result)


