from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_regression_metrics
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ppg import PPGDataset
from timeseries.models.chronos import ChronosModel
from timeseries.decoder.regression.ridge import RidgeDecoder

device='cuda:0'

task_cfg={
    'task_type': 'regression',
    'inference_config': {
        'batch_size': 32,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 32,
        'shuffle':False
    },
    'label': 'hr',
}  
dataset_cfg={
        'dataset_path': '../../dataset/PPG-data',
      
} 


dataloader_train = DataLoader(PPGDataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])


P=Pipeline(ChronosModel(device,'amazon/chronos-t5-tiny'))
mlp_decoder=P.add_decoder(RidgeDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_regression_metrics(y_test, y_pred)
print(result['mae'])






