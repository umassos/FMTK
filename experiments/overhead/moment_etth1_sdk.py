from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_forecasting_metrics
from torch.utils.data import DataLoader,TensorDataset
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.models.moment import MomentModel
from timeseries.decoder.forecasting.mlp import MLPDecoder
from timeseries.logger import Logger
from tqdm import tqdm
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
        'dataset_path': '../../../dataset/ETTh1',
} 


dataloader_train = DataLoader(ETTh1Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

momentlogger=Logger(device,'moment_etth1_sdk')
with (momentlogger.measure("load backbone", device=momentlogger.device) if momentlogger else nullcontext()):
    P=Pipeline(MomentModel(device,'AutonLab/MOMENT-1-base'),momentlogger)
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':64*768,'output_dim':192,'dropout':0.1}),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])

path = momentlogger.save()
print("saved:", path)


