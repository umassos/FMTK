from timeseries.metrics import get_mae
from timeseries.pipeline import Pipeline
from timeseries.datasets.ppg import PPGDataset
from timeseries.components.backbones.moment import MomentModel
from timeseries.components.decoders.regression.mlp import MLPDecoder
from timeseries.components.encoders.diff import LinearChannelCombiner
from timeseries.logger import Logger
from torch.utils.data import DataLoader
from peft import LoraConfig

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
        'dataset_path': '../../../dataset/PPG-data',      
} 
lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05)



dataloader_train = DataLoader(PPGDataset(dataset_cfg,task_cfg,split='train'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_val = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='val'), batch_size=task_cfg['train_config']['batch_size'], shuffle=task_cfg['train_config']['shuffle'])
dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='test') , batch_size=task_cfg['inference_config']['batch_size'], shuffle=task_cfg['inference_config']['shuffle'])

model='AutonLab/MOMENT-1-large'



# momentlogger=Logger(device,'moment_ppg_mlp_d')
# P=Pipeline(MomentModel(device,model),momentlogger)
# mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
# P.train(dataloader_train,parts_to_train=['decoder'],cfg=task_cfg['train_config'])
# y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
# result=get_regression_metrics(y_test, y_pred)
# r_d=result['mae']
# path = momentlogger.save()
# print("saved:", path)

# momentlogger=Logger(device,'moment_ppg_mlp_d_e')
# P=Pipeline(MomentModel(device,model),momentlogger)
# mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
# encoder=P.add_encoder(LinearChannelCombiner(num_channels=3,new_num_channels=1),load=True)
# P.train(dataloader_train,parts_to_train=['encoder','decoder'],cfg=task_cfg['train_config'])
# y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
# result=get_regression_metrics(y_test, y_pred)
# r_d_e=result['mae']
# path = momentlogger.save()
# print("saved:", path)

momentlogger=Logger(device,'moment_ppg_mlp_d_a')
P=Pipeline(MomentModel(device,'large'),momentlogger)
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128}),load=True)
peft_adapter=P.add_adapter(lora_config)
P.train(dataloader_train,parts_to_train=['decoder','adapter'],cfg=task_cfg['train_config'])
y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
result=get_mae(y_test, y_pred)
r_d_a=result['mae']
print(r_d_a)
path = momentlogger.save()
print("saved:", path)

# momentlogger=Logger(device,'moment_ppg_mlp_d_e_a')
# P=Pipeline(MomentModel(device,model),logger=momentlogger)
# mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':1024,'output_dim':1,'hidden_dim':128},lr=task_cfg['train_config']['lr']),load=True)
# encoder=P.add_encoder(LinearChannelCombiner(num_channels=3,new_num_channels=1),load=True)
# peft_adapter=P.add_adapter(lora_config)
# P.train(dataloader_train,parts_to_train=['encoder','decoder','adapter'],cfg=task_cfg['train_config'])
# y_test,y_pred=P.predict(dataloader_test,cfg=task_cfg['inference_config'])
# result=get_regression_metrics(y_test, y_pred)
# r_d_e_a=result['mae']
# print(r_d_e_a)
# path = momentlogger.save()
# print("saved:", path)


print(r_d)
print(r_d_a)
print(r_d_e)
print(r_d_e_a)


# print("inference time")
# print(i_d)
# print(i_d_a)
# print(i_d_e)
# print(i_d_e_a)






