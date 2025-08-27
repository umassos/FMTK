from fmtk.pipeline import Pipeline
from fmtk.datasets.ecg5000 import ECG5000Dataset
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.classification.mlp import MLPDecoder
from fmtk.components.decoders.classification.svm import SVMDecoder
from fmtk.logger import Logger
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader

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
dataloader_val = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

P=Pipeline(ChronosModel(device,'large'))
svm_decoder=P.add_decoder(SVMDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_accuracy(y_test, y_pred)
print(result)



