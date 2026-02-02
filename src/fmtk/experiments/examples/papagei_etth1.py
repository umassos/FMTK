from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_forecasting_metrics
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.models.papagei import PapageiModel
from timeseries.decoder.forecasting.mlp import MLPDecoder

device='cuda:0'
model_cfg={
    'in_channels':1, 
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,
    }

task_cfg={
    'task_type': 'forecasting',
    'inference_config': {
        'batch_size': 8,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 8,
        'shuffle':False
    }
}  
dataset_cfg={
        'dataset_path': '../../dataset/ETTh1',
} 


dataloader_train = DataLoader(ETTh1Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

P=Pipeline(PapageiModel(device,model_name='papagei_p',model_config=model_cfg))
mlp_decoder=P.add_decoder(MLPDecoder(device),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_forecasting_metrics(y_test, y_pred)
print(result.mae)


