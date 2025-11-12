from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_classification_metrics
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.components.backbones.papagei import PapageiModel
from timeseries.components.decoders.classification.mlp import MLPDecoder
from timeseries.components.decoders.classification.svm import SVMDecoder
from timeseries.components.decoders.classification.knn import KNNDecoder
from timeseries.components.decoders.classification.logisticregression import LogisticRegressionDecoder
from timeseries.components.decoders.classification.randomforest import RandomForestDecoder

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
        'dataset_path': '../../../dataset/ECG5000',
} 


dataloader_train = DataLoader(ECG5000Dataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])


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
svm_decoder=P.add_decoder(SVMDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result_svm=get_classification_metrics(y_test, y_pred)

P=Pipeline(PapageiModel(device,model_name='papagei_s',model_config=model_cfg))
knn_decoder=P.add_decoder(KNNDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result_knn=get_classification_metrics(y_test, y_pred)

P=Pipeline(PapageiModel(device,model_name='papagei_s',model_config=model_cfg))
lr_decoder=P.add_decoder(LogisticRegressionDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result_lr=get_classification_metrics(y_test, y_pred)

P=Pipeline(PapageiModel(device,model_name='papagei_s',model_config=model_cfg))
rf_decoder=P.add_decoder(RandomForestDecoder(),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result_rf=get_classification_metrics(y_test, y_pred)

P=Pipeline(PapageiModel(device,model_name='papagei_s',model_config=model_cfg))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':512,'output_dim':5,'hidden_dim':128}),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result_mlp=get_classification_metrics(y_test, y_pred)

print(result_svm['accuracy'])
print(result_knn['accuracy'])
print(result_lr['accuracy'])
print(result_rf['accuracy'])
print(result_mlp['accuracy'])

