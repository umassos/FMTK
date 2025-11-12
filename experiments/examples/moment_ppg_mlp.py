from sklearn.model_selection import train_test_split, GridSearchCV
from timeseries.metrics import get_regression_metrics
from torch.utils.data import DataLoader
import psutil, os
from timeseries.pipeline import Pipeline
from timeseries.datasets.ppg import PPGDataset
from timeseries.models.moment import MomentModel
from timeseries.decoder.regression.mlp import MLPDecoder
from timeseries.encoders.diff import LinearChannelCombiner
from peft import LoraConfig
import numpy as np
import time
device='cuda:0'

task_cfg={
    'task_type': 'regression',
    'inference_config': {
        'batch_size': 32,
        'shuffle':False,
        },    
    'train_config':{
        'batch_size': 32,
        'shuffle':False,
        'epochs':50,
        'lr':1e-2
    },
    'label': 'hr',
    
}  
dataset_cfg={
        'dataset_path': '../../dataset/PPG-data',      
} 
lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,)



dataloader_train = DataLoader(PPGDataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

model='AutonLab/MOMENT-1-large'

P=Pipeline(MomentModel(device,model))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
encoder=P.add_encoder(LinearChannelCombiner(num_channels=3,new_num_channels=1),load=True)
peft_adapter=P.add_adapter(lora_config)
train_st=time.time()
P.train(dataloader_train,parts_to_train=['encoder','decoder','adapter'],cfg=task_cfg['train_config'])
inf_st=time.time()
y_test,y_pred,times=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
inf_et=time.time()
result=get_regression_metrics(y_test, y_pred)
r_d_e_a=result['mae']
t_d_e_a=inf_st-train_st
i_d_e_a=np.mean(times)
total_d_e_a=inf_et-inf_st
print(r_d_e_a)
print(t_d_e_a)
print(i_d_e_a)
print(total_d_e_a)

P=Pipeline(MomentModel(device,model))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
train_st=time.time()
P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
inf_st=time.time()
y_test,y_pred,times=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
inf_et=time.time()
result=get_regression_metrics(y_test, y_pred)
r_d=result['mae']
t_d=inf_st-train_st
i_d=np.mean(times)
total_d=inf_et-inf_st
print(r_d)
print(t_d)
print(i_d)
print(total_d)

P=Pipeline(MomentModel(device,model))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
encoder=P.add_encoder(LinearChannelCombiner(num_channels=3,new_num_channels=1),load=True)
train_st=time.time()
P.train(dataloader_train,parts_to_train=['encoder','decoder'],cfg=task_cfg['train_config'])
inf_st=time.time()
y_test,y_pred,times=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
inf_et=time.time()
result=get_regression_metrics(y_test, y_pred)
r_d_e=result['mae']
t_d_e=inf_st-train_st
i_d_e=np.mean(times)
total_d_e=inf_et-inf_st
print(r_d_e)
print(t_d_e)
print(i_d_e)
print(total_d_e)


P=Pipeline(MomentModel(device,model))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
peft_adapter=P.add_adapter(lora_config)
train_st=time.time()
P.train(dataloader_train,parts_to_train=['decoder','adapter'],cfg=task_cfg['train_config'])
inf_st=time.time()
y_test,y_pred,times=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
inf_et=time.time()
result=get_regression_metrics(y_test, y_pred)
r_d_a=result['mae']
t_d_a=inf_st-train_st
i_d_a=np.mean(times)
total_d_a=inf_et-inf_st
print(r_d_a)
print(t_d_a)
print(i_d_a)
print(total_d_a)



# print(r_d)
# print(r_d_a)
# print(r_d_e)
# print(r_d_e_a)

# print("train time")
# print(t_d)
# print(t_d_a)
# print(t_d_e)
# print(t_d_e_a)

# print("inference time")
# print(i_d)
# print(i_d_a)
# print(i_d_e)
# print(i_d_e_a)






