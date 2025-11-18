from component_loader import get_model_class, get_decoder_class, get_encoder_class,get_adapter_class
from dataset_loader import get_dataset_class
from sklearn.model_selection import train_test_split, GridSearchCV
from fmtk.metrics import get_mae, get_accuracy
from torch.utils.data import DataLoader,ConcatDataset
import psutil, os
from fmtk.pipeline import Pipeline
from config import *
from fmtk.metrics import get_mae, get_accuracy
from fmtk.utils import control_randomness
from fmtk.logger import Logger
import csv
import json

class InferencePipeline:
    def __init__(self,task_name,task_info, pipeline):

        self.backbone_cfg = backbones[pipeline['backbone']]
        self.dataset_cfg = datasets[task_info['datasets'][0]]
        self.train=task_info['train']
        self.task_cfg = task_info
        self.task_name=task_name
        self.pipeline=pipeline
        self.model_name = self.backbone_cfg['model_name']
        self.model_type = self.backbone_cfg['model_type']
        self.device = device
        control_randomness(13)

        dataset_class = get_dataset_class(self.dataset_cfg['dataset_type'])
        self.dataset_instance_train = dataset_class(self.dataset_cfg, self.task_cfg,split='train')
        self.dataset_instance_test = dataset_class(self.dataset_cfg, self.task_cfg,split='test')
        self.dataset_instance_val = dataset_class(self.dataset_cfg, self.task_cfg,split='val')
        self.dataloader_train = DataLoader(self.dataset_instance_train, batch_size=self.task_cfg['train_config']['batch_size'], shuffle=self.task_cfg['train_config']['shuffle'])
        self.dataloader_val = DataLoader(self.dataset_instance_val, batch_size=self.task_cfg['inference_config']['batch_size'], shuffle=self.task_cfg['inference_config']['shuffle'])
        self.dataloader_test = DataLoader(self.dataset_instance_test, batch_size=self.task_cfg['inference_config']['batch_size'], shuffle=self.task_cfg['inference_config']['shuffle'])

        # combined_dataset = ConcatDataset([self.dataset_instance_test,self.dataset_instance_val])

        # self.dataloader_test = DataLoader(
        #     combined_dataset,
        #     batch_size=self.task_cfg['inference_config']['batch_size'],
        #     shuffle=self.task_cfg['inference_config']['shuffle']
        # )

    def run(self):
        print(f"Running inference for model: {self.model_type}_{self.model_name} on dataset: {self.dataset_cfg['dataset_type']}")
        backbone_class = get_model_class(self.backbone_cfg['model_type'])
        logger=Logger(device,'log')
        with (logger.measure("backbone", device=logger.device) if logger else nullcontext()):
            fm_instance = backbone_class(self.device,self.model_name,self.backbone_cfg.get('model_config',None))

        P=Pipeline(fm_instance,logger=logger)
        for path in self.pipeline['paths']:
            if 'decoder' in path:
                decoder_class = get_decoder_class(self.task_cfg['task_type'],decoders[path['decoder']]['decoder_type'])
                with (logger.measure("decoder", device=logger.device) if logger else nullcontext()):
                    if 'decoder_config' in decoders[path['decoder']]:
                        decoder_instance = decoder_class(**decoders[path['decoder']]['decoder_config'])
                    else:
                        decoder_instance = decoder_class()

                    P.add_decoder(decoder_instance,load=True,train=self.train,path=path['path'])
            else:
                P.unload_decoder()     

            if 'encoder' in path:
                encoder_class = get_encoder_class(encoders[path['encoder']]['encoder_type'])
                if 'encoder_config' in encoders[path['encoder']]:
                    encoder_instance = encoder_class(**encoders[path['encoder']]['encoder_config'])
                else:
                    encoder_instance = encoder_class()
                P.add_encoder(encoder_instance,load=True)
            else:
                P.unload_encoder()

            if 'adapter' in path:
                adapter_class = get_adapter_class(adapters[path['adapter']]['adapter_type'])
                if 'adapter_config' in adapters[path['adapter']]:
                    adapter_instance = adapter_class(**adapters[path['adapter']]['adapter_config'])
                else:
                    adapter_instance = adapter_class()
                P.add_adapter(adapter_instance)
            else:
                P.unload_adapter()
            if self.train:
                P.train(self.dataloader_train,parts_to_train=path['parts_to_train'],cfg=self.task_cfg['train_config'],path=path['path'])
                print("Training complete")     
            y_test,y_pred=P.predict(self.dataloader_test,cfg=self.task_cfg['inference_config'])   
            summary=logger.summary()
            if not self.train:
                base_dir = os.path.dirname(__file__)
                with open(f"{base_dir}/../../src/fmtk/saved/{path['path']}/pipeline.json", 'r') as file:
                    data = json.load(file)
                summary.update({'train':data['train']})

            if self.task_cfg['task_type']=='regression' or self.task_cfg['task_type']=='forecasting':
                    metrics = {
                            "backbone": self.pipeline['backbone'],
                            "decoder": path.get('decoder',None),
                            "encoder": path.get('encoder',None),
                            "adapter": path.get('adapter',None),
                            "dataset_name": self.task_cfg['datasets'][0],
                            "device": device ,
                            "task_name": self.task_name,
                            "metric":'mae',
                            "result": get_mae(y_test, y_pred),
                            "backbone memory": summary['backbone']['gpu peak'],
                            "decoder memory": summary['decoder']['gpu peak'],
                            "train time":summary['train']['gpu time'],
                            "train mem peak":summary['train']['gpu peak'],
                            "train energy":summary['train']['gpu energy'],
                            "inference time":summary['predict']['gpu time'],
                            "inference mem peak":summary['predict']['gpu peak'],
                            "inference energy":summary['predict']['gpu energy'],
                            } 
            elif self.task_cfg['task_type']=='classification':
                metrics = {
                        "backbone": self.pipeline['backbone'],
                        "decoder": path.get('decoder',None),
                        "encoder": path.get('encoder',None),
                        "adapter": path.get('adapter',None),
                        "dataset_name": self.task_cfg['datasets'][0],
                        "device": device ,
                        "task_name":self.task_name ,
                        "metric":'accuracy',
                        "result": get_accuracy(y_test, y_pred),
                        "backbone memory": summary['backbone']['gpu peak'],
                        "decoder memory": summary['decoder']['gpu peak'],
                        "train time":summary['train']['gpu time'],
                        "train mem peak":summary['train']['gpu peak'],
                        "train energy":summary['train']['gpu energy'],
                        "inference time":summary['predict']['gpu time'],
                        "inference mem peak":summary['predict']['gpu peak'],
                        "inference energy":summary['predict']['gpu energy'],
                        } 
                        
            write_header = not os.path.exists(log_file)
            with open(log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics)  
         

    

    


