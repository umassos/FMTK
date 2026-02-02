from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_regression_metrics
from torch.utils.data import DataLoader, ConcatDataset
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ppg import PPGDataset
from timeseries.components.backbones.papagei import PapageiModel
from timeseries.components.decoders.regression.ridge import RidgeDecoder

device='cuda:0'

task_cfg={
    'task_type': 'regression',
    'inference_config': {
        'batch_size': 256,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 256,
        'shuffle':False
    },
    'label': 'diasbp',
}  
dataset_cfg={
        'dataset_path': '../../../dataset/PPG-data',
        'dataset_type': 'PPG-data',
        'case_name': 'subject_ID',        
} 

dataloader_train = DataLoader(PPGDataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
val_dataset  = PPGDataset(dataset_cfg, task_cfg, split='val')
test_dataset = PPGDataset(dataset_cfg, task_cfg, split='test')

combined_dataset = ConcatDataset([test_dataset,val_dataset])

dataloader_test = DataLoader(
    combined_dataset,
    batch_size=task_cfg['inference_config']['batch_size'],
    shuffle=task_cfg['inference_config']['shuffle']
)

model_cfg={
    'in_channels':1, 
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,
    }
P=Pipeline(PapageiModel(device,model_name='papagei_p',model_config=model_cfg))
ridge_decoder=P.add_decoder(RidgeDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
results_papagei_p_sysbp=get_regression_metrics(y_test, y_pred)

model_cfg={'in_channels':1,
        'base_filters': 32,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 18,
        'n_classes': 512,
        'n_experts': 3
        }

P=Pipeline(PapageiModel(device,model_name='papagei_s',model_config=model_cfg))
ridge_decoder=P.add_decoder(RidgeDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
results_papagei_s_sysbp=get_regression_metrics(y_test, y_pred)

model_cfg={'in_channels':1,
            'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
        }
P=Pipeline(PapageiModel(device,model_name='papagei_s_svri',model_config=model_cfg))
ridge_decoder=P.add_decoder(RidgeDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
results_papagei_s_svri_sysbp=get_regression_metrics(y_test, y_pred)

print(f"PaPaGei-S Systolic BP MAE: {results_papagei_s_sysbp['mae']}")
print(f"PaPaGei-P Systolic BP MAE: {results_papagei_p_sysbp['mae']}")
print(f"PaPaGei-S sVRI Systolic BP MAE: {results_papagei_s_svri_sysbp['mae']}")