from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_classification_metrics
from torch.utils.data import DataLoader, TensorDataset
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.models.moment import MomentModel
from timeseries.decoder.classification.svm import SVMDecoder
from timeseries.logger import Logger
from tqdm import tqdm

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
    }
}  
dataset_cfg={
        'dataset_path': '../../../dataset/ECG5000',
} 


dataloader_train = DataLoader(ECG5000Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

momentlogger=Logger(device,'moment_ecg_sdk')
with (momentlogger.measure("load backbone", device=momentlogger.device) if momentlogger else nullcontext()):
    P=Pipeline(MomentModel(device,'AutonLab/MOMENT-1-base'),logger=momentlogger)
svm_decoder=P.add_decoder(SVMDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])

for batch_x, batch_masks, batch_labels in tqdm(dataloader_test, total=len(dataloader_test)):
    single_dataset = TensorDataset(batch_x, batch_masks, batch_labels)
    dataloader = DataLoader(single_dataset, batch_size=len(batch_x))
    y_test,y_pred=P.predict(dataloader,cfg=task_cfg['inference_config'])

path = momentlogger.save()
print("saved:", path)

