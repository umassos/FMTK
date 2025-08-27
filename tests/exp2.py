from sklearn.model_selection import train_test_split, GridSearchCV
from fmtk.metrics import get_mae
from torch.utils.data import DataLoader
import psutil, os
from fmtk.pipeline import Pipeline
from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.decoders.forecasting.mlp import MLPDecoder
from sklearn.metrics import mean_absolute_error
device='cuda:0'

task_cfg={
    'task_type': 'forecasting',
    'inference_config': {
        'batch_size': 8,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 8,
        'shuffle':False,
        'epochs':1,
        'lr':1e-4

    }
}  
dataset_cfg={
        'dataset_path': '../dataset/ETTh1',
} 


dataloader_train = DataLoader(ETTh1Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

P=Pipeline(MomentModel(device,'base'))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':64*768,'output_dim':192,'dropout':0.1}),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_mae(y_test, y_pred)
print(result)




