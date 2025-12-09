from fmtk.pipeline import Pipeline
from fmtk.datasets.ppg import PPGDataset
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.decoders.regression.mlp import MLPDecoder
from fmtk.components.encoders.diff import LinearChannelCombiner
from fmtk.metrics import get_mae
from torch.utils.data import DataLoader
from peft import LoraConfig

device='cuda:0'

task_cfg={'task_type': 'regression','label': 'hr'}  
# 'hr' for heart rate, 'sysbp' for systolic blood pressure, 'diasbp' for diastolic blood pressure
train_config={'batch_size': 32,'shuffle':False,'epochs':50,'lr':1e-2}
inference_config= {'batch_size': 32,'shuffle':False}  
dataset_cfg={'dataset_path': '../dataset/PPG-data'} 

lora_config = LoraConfig(r=64,lora_alpha=32,target_modules=["q", "v"],lora_dropout=0.05)

dataloader_train = DataLoader(PPGDataset(dataset_cfg,task_cfg,split='train'), batch_size=train_config['batch_size'], shuffle=train_config['shuffle'])
dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg,split='test') , batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

P=Pipeline(MomentModel(device,'base'))
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':768,'output_dim':1,'hidden_dim':128}),load=True)
encoder=P.add_encoder(LinearChannelCombiner(num_channels=3,new_num_channels=1),load=True)
peft_adapter=P.add_adapter(lora_config)
P.train(dataloader_train,parts_to_train=['encoder','decoder','adapter'],cfg=train_config)
y_test,y_pred=P.predict(dataloader_test,cfg=inference_config)
result=get_mae(y_test, y_pred)
print(result)







