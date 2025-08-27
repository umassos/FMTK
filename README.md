# FMTK: A Modular Toolkit for Composable Time Series Foundation Model Pipelines

## Overview
Foundation models (FMs) have opened new avenues for machine learning applications due to their ability to adapt to new and unseen tasks with minimal or no further training. Time-series foundation models (TSFMs)---FMs trained on time-series data---have shown strong performance on classification, regression, and imputation tasks. Recent pipelines combine TSFMs with task-specific encoders, decoders, and adapters to improve performance; however, assembling such pipelines typically requires ad hoc, model-specific implementations that hinder modularity and reproducibility. We introduce FMTK, an open-source, lightweight and extensible toolkit for constructing and fine-tuning TSFM pipelines via standardized backbone and component abstractions. FMTK enables flexible composition across models and tasks, achieving correctness and performance with an average of seven lines of code.

![Architecture](./images/architecture.jpg)

## How to use FMTK
### Zero shot use
1. Install fmtk package
```
pip install fmtk
```
### Developer
```
git clone https://github.com/umassos/fmtk.git
cd FMTK
conda create -n fmtk python=3.10
conda activate fmtk
pip install -e
```

2. Import libraries
```
from fmtk.pipeline import Pipeline
from fmtk.datasets.ecg5000 import ECG5000Dataset
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.classification.mlp import MLPDecoder
from fmtk.components.decoders.classification.svm import SVMDecoder
from fmtk.logger import Logger
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader
```
3. Set device and Dataloader
```
device='cuda:0'

task_cfg={
    'task_type': 'classification',
    'inference_config': {
        'batch_size': 32,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 32,
        'shuffle':False,
        'epochs':50,
        'lr': 1e-2

    }
}  
dataset_cfg={
        'dataset_path': '../dataset/ECG5000',
} 


dataloader_train = DataLoader(ECG5000Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])
```

4. Use Pipeline 
```
P=Pipeline(ChronosModel(device,'large'))
svm_decoder=P.add_decoder(SVMDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_accuracy(y_test, y_pred)
```

