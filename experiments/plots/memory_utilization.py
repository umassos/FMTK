from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_classification_metrics
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.datasets.ppg import PPGDataset
from timeseries.models.moment import MomentModel
from timeseries.decoder.classification.mlp import MLPDecoder
import time
def get_gpu_memory():
    handle = nvmlDeviceGetHandleByIndex(0) 
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / 1024**2

# Initialize NVML once
nvmlInit()

device='cuda:0'

cls_task_cfg={
    'task_type': 'classification',
    'inference_config': {
        'batch_size': 32,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 32,
        'shuffle':False,
        'epochs':1,
        'lr':1e-2
    }
}  
dataset_cfg={
        'dataset_path': '../../dataset/ECG5000',
} 


dataloader_train_cls = DataLoader(ECG5000Dataset(dataset_cfg,cls_task_cfg,split='train'), batch_size=cls_task_cfg['train_config']['batch_size'], shuffle=cls_task_cfg['train_config']['shuffle'])
dataloader_val_cls = DataLoader(ECG5000Dataset(dataset_cfg, cls_task_cfg,split='val'), batch_size=cls_task_cfg['train_config']['batch_size'], shuffle=cls_task_cfg['train_config']['shuffle'])
dataloader_test_cls = DataLoader(ECG5000Dataset(dataset_cfg, cls_task_cfg,split='test') , batch_size=cls_task_cfg['inference_config']['batch_size'], shuffle=cls_task_cfg['inference_config']['shuffle'])

reg_task_cfg={
    'task_type': 'regression',
    'inference_config': {
        'batch_size': 32,
        'shuffle':False,
        },    
    'train_config':{
        'batch_size': 32,
        'shuffle':False,
        'epochs':50,
        'lr':1e-2
    },
    'label': 'hr',
    
}  
dataset_cfg={
        'dataset_path': '../../dataset/PPG-data',      
} 

dataloader_train_reg = DataLoader(PPGDataset(dataset_cfg,reg_task_cfg,split='train'), batch_size=reg_task_cfg['train_config']['batch_size'], shuffle=reg_task_cfg['train_config']['shuffle'])
dataloader_val_reg = DataLoader(PPGDataset(dataset_cfg, reg_task_cfg,split='val'), batch_size=reg_task_cfg['train_config']['batch_size'], shuffle=reg_task_cfg['train_config']['shuffle'])
dataloader_test_reg = DataLoader(PPGDataset(dataset_cfg, reg_task_cfg,split='test') , batch_size=reg_task_cfg['inference_config']['batch_size'], shuffle=reg_task_cfg['inference_config']['shuffle'])

memory=[]
parameters=[]
s=get_gpu_memory()
P=Pipeline(MomentModel(device,'AutonLab/MOMENT-1-large'))
e=get_gpu_memory()
memory.append(e-s)
parameters.append(sum(p.numel() for p in P.model_instance.model.parameters()))
torch.cuda.synchronize()
print("alloc=", torch.cuda.memory_allocated()/1e6,
      "reserved=", torch.cuda.memory_reserved()/1e6)

mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train_cls,parts_to_train=['decoder'],cfg=cls_task_cfg['train_config'])
e=get_gpu_memory()
memory.append(e-s)
parameters.append(sum(p.numel() for p in P.active_decoder.model.parameters())+parameters[-1])
torch.cuda.synchronize()
print("alloc=", torch.cuda.memory_allocated()/1e6,
      "reserved=", torch.cuda.memory_reserved()/1e6)

mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train_cls,parts_to_train=['decoder'],cfg=cls_task_cfg['train_config'])
e=get_gpu_memory()
memory.append(e-s)
parameters.append(sum(p.numel() for p in P.active_decoder.model.parameters())+parameters[-1])
torch.cuda.synchronize()
print("alloc=", torch.cuda.memory_allocated()/1e6,
      "reserved=", torch.cuda.memory_reserved()/1e6)

mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train_cls,parts_to_train=['decoder'],cfg=cls_task_cfg['train_config'])
e=get_gpu_memory()
memory.append(e-s)
parameters.append(sum(p.numel() for p in P.active_decoder.model.parameters())+parameters[-1])
torch.cuda.synchronize()
print("alloc=", torch.cuda.memory_allocated()/1e6,
      "reserved=", torch.cuda.memory_reserved()/1e6)

mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train_cls,parts_to_train=['decoder'],cfg=cls_task_cfg['train_config'])
e=get_gpu_memory()
memory.append(e-s)
parameters.append(sum(p.numel() for p in P.active_decoder.model.parameters())+parameters[-1])
torch.cuda.synchronize()
print("alloc=", torch.cuda.memory_allocated()/1e6,
      "reserved=", torch.cuda.memory_reserved()/1e6)

mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train_cls,parts_to_train=['decoder'],cfg=cls_task_cfg['train_config'])
e=get_gpu_memory()
memory.append(e-s)
parameters.append(sum(p.numel() for p in P.active_decoder.model.parameters())+parameters[-1])
torch.cuda.synchronize()
print("alloc=", torch.cuda.memory_allocated()/1e6,
      "reserved=", torch.cuda.memory_reserved()/1e6)

print(memory)
print(parameters)


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tasks=range(0,6)


x = np.arange(len(tasks))
width = 0.4

fig, ax1 = plt.subplots(figsize=(8, 4))

# Left axis: inference_time bars (shifted left)
b1 = ax1.bar(x - width/2, memory, width, label='Memory',color='#faa275')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks)
ax1.set_ylabel('Memory (MB)')
ax1.set_ylim(1600,2200)
ax1.set_xlabel('#Tasks')

# Right axis: MAE bars (shifted right)
ax2 = ax1.twinx()
b2 = ax2.bar(x + width/2, parameters, width, label='Paramerters',color='#ce6a85')
ax2.set_ylabel('Paramerters')
ax2.set_ylim(341200000,341900000)
# One legend for both axes
handles = [b1, b2]
tasks_legend = [h.get_label() for h in handles]
ax1.legend(handles, tasks_legend,bbox_to_anchor=(0.5, 1.19),loc='upper center',ncol=2)

plt.tight_layout()
sns.despine(fig, right=False, top=True)
plt.savefig("memory.pdf")
plt.show()