from fmtk.pipeline import Pipeline
from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.components.backbones.papagei import PapageiModel
from fmtk.components.decoders.forecasting.mlp import MLPDecoder
from fmtk.metrics import get_mae
from torch.utils.data import DataLoader


device='cuda:0'

task_cfg={
    'task_type': 'forecasting',

}  
inference_config= {
        'batch_size': 8,
        'shuffle':False
        }    
train_config={
        'batch_size': 8,
        'shuffle':False,
        'epochs':1,
        'lr':1e-4}
dataset_cfg={
        'dataset_path': '../dataset/ETTh1',
} 

dataloader_train = DataLoader(ETTh1Dataset(dataset_cfg,task_cfg,split='train'), batch_size=train_config['batch_size'], shuffle=train_config['shuffle'])
dataloader_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg,split='test') , batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

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
mlp_decoder=P.add_decoder(MLPDecoder(device,cfg={'input_dim':512,'output_dim':192,'dropout':0.1}),load=True)
P.train(dataloader_train,parts_to_train=['decoder'],cfg=train_config)
y_test,y_pred=P.predict(dataloader_test,cfg=inference_config)
result=get_mae(y_test, y_pred)
print(result)
