from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_mae
from torch.utils.data import DataLoader,ConcatDataset
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ppg import PPGDataset
from timeseries.components.backbones.moment import MomentModel
from timeseries.components.decoders.regression.ridge import RidgeDecoder
from timeseries.components.decoders.regression.mlp import MLPDecoder


device='cuda:0'

task_cfg={
    'task_type': 'regression',
    'inference_config': {
        'batch_size': 256,
        'shuffle':False
        },    
    'train_config':{
        'batch_size': 256,
        'shuffle':False,
        'epochs':50,
        'lr':1e-2,
    },
    'label': 'sysbp',
}  
dataset_cfg={
        'dataset_path': '../../../dataset/PPG-data',
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

# P=Pipeline(MomentModel(device,'large'))
# mlp_decoder=P.add_decoder(RidgeDecoder(),load=True)
# P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
# y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
# result=get_regression_metrics(y_test, y_pred)
# print(result['mae'])

P=Pipeline(MomentModel(device,'large'))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_mae(y_test, y_pred)
print(result)






