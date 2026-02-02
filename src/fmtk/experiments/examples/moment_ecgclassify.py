from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_classification_metrics
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.models.moment import MomentModel
from timeseries.decoder.classification.mlp import MLPDecoder

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
        'lr':1e-2
    }
}  
dataset_cfg={
        'dataset_path': '../../dataset/ECG5000',
} 


dataloader_train = DataLoader(ECG5000Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

P=Pipeline(MomentModel(device,'AutonLab/MOMENT-1-base'))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':768,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_classification_metrics(y_test, y_pred)
print(result['accuracy'])


