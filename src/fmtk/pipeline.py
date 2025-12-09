import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import time
import os
from contextlib import nullcontext
import json

class Pipeline:
    def __init__(self, model_instance,logger=None):
        self.logger=logger
        self.model_instance = model_instance
        self.decoders = {}
        self.encoders = {}
        self.active_decoder = None
        self.active_encoder = None
        self.decoder_id=0
        self.adapter_id=0
        self.encoder_id=0
        self.base_dir = os.path.dirname(__file__)
        
    def add_adapter(self,peft_cfg):
        adapter_name=f'adapter_{self.adapter_id}'
        self.adapter_id+=1
        if self.model_instance.peft_enable:
            with (self.logger.measure("add_adapter", device=self.logger.device) if self.logger else nullcontext()):
                self.model_instance.model.add_adapter(adapter_name=adapter_name, peft_config=peft_cfg)
            return adapter_name
        else:
            self.model_instance.enable_peft(peft_cfg)
            return 'default'
        
    def unload_adapter(self):
        if hasattr(self.model_instance,'peft_enable') and self.model_instance.peft_enable:
            self.model_instance.disable_adapters()



    def add_encoder(self,encoder_obj,load=True):
        encoder_name=f'encoder_{self.encoder_id}'
        with (self.logger.measure("add_encoder", device=self.logger.device) if self.logger else nullcontext()):
            self.encoders[encoder_name] = encoder_obj
        self.encoder_id+=1
        if load:
            self.active_encoder = self.encoders[encoder_name]
        return f"{encoder_name}"
    
    def unload_encoder(self):
        self.active_encoder=None
    
    def add_decoder(self,decoder_obj,load=True,train=True,path=None):
        """Adds a named decoder to the manager."""
        decoder_name=f"decoder_{self.decoder_id}"
        with (self.logger.measure("add_decoder", device=self.logger.device) if self.logger else nullcontext()):
            if not train:
                self.decoders[decoder_name]= decoder_obj
                self.decoders[decoder_name].model.load_state_dict(torch.load(f"{self.base_dir}/saved/{path}/decoder.pth"))
                
            else:
                self.decoders[decoder_name] = decoder_obj
            self.decoder_id+=1
            if load:
                if hasattr(decoder_obj,'to_device'):
                    decoder_obj.to_device()
                self.active_decoder = self.decoders[decoder_name]
        return f"{decoder_name}"

    def load_decoder(self,decoder_id,swap=False):
        """Sets the active decoder for future predict/train."""        
        if decoder_id not in self.decoders:
            raise ValueError(f"decoder {decoder_id} not found. Available: {list(self.decoders.keys())}")
        with (self.logger.measure("load_decoder", device=self.logger.device) if self.logger else nullcontext()):
            if swap:
                if self.active_decoder is not None:
                    self.active_decoder.to_cpu()
                self.decoders[decoder_id].to_device()
                
            self.active_decoder = self.decoders[decoder_id]

    def unload_decoder(self):
        self.active_decoder=None

    def train(self, train_loader, val_loader=None, parts_to_train=['decoder'],cfg=None,path=None):
        trains_decoder = 'decoder' in parts_to_train
        trains_adapter = 'adapter' in parts_to_train
        trains_encoder = 'encoder' in parts_to_train

        param_groups = []
        if self.active_encoder is not None:
            if trains_encoder:
                if hasattr(self.active_encoder,'fit'):
                    train_loader=self.active_encoder.fit(train_loader)
                else:
                    encoder_params = list(self.active_encoder.trainable_parameters())
                    if len(encoder_params):
                        param_groups.append({"params": encoder_params, "lr": cfg['lr']})
                    if hasattr(self.active_decoder,'fit'):
                        "Has own fit non differentiable"
                        raise ValueError("Need differentiable decoder as attached encoder. Because how will backward propagation happen")
            else:
                train_loader = self._encoder_loader(train_loader, cfg)
        if trains_adapter:
            adapter_params = list(self.model_instance.adapter_trainable_parameters())
            param_groups.append({"params": adapter_params, "lr": cfg['lr']})
            if hasattr(self.active_decoder,'fit'):
                "Has own fit non differentiable"
                raise ValueError("Need differentiable decoder as attached adapter. Because how will backward propagation happen")
        if trains_decoder:
            if hasattr(self.active_decoder,'fit'):
                print("[Trainer] Extracting test embeddings...")
                with (self.logger.measure("train", device=self.logger.device) if self.logger else nullcontext()):
                    train_loader = self._embed_loader(train_loader, cfg)
                    if hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model:
                        self.active_decoder.fit(self.model_instance.model, train_loader,cfg)
                        return
                    else:
                        print("Finetuning decoder with own fit")
                        self.active_decoder.fit(train_loader,cfg)
                        return
            else:
                dec_params = list(self.active_decoder.trainable_parameters())
                if len(dec_params):
                    param_groups.append({"params": dec_params, "lr": cfg['lr']})   

        with (self.logger.measure("train", device=self.logger.device) if self.logger else nullcontext()):
            optimizer = torch.optim.Adam(param_groups)
            criterion = getattr(self.active_decoder, "criterion")
            for _ in range(cfg['epochs']):
                for batch in tqdm(train_loader):
                        optimizer.zero_grad()
                        if len(batch)==3:
                            x, mask, y = batch["x"], batch["mask"], batch["y"]
                        else:
                            x, y = batch["x"], batch["y"]
                            mask=None
            
                        logits=self.forward(x,mask)
                        if (hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model and hasattr(self.model_instance.model, "normalizer")):
                            logits = self.model_instance.model.normalizer(x=logits, mode="denorm")
                        if isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)): 
                            logits = logits.float()
                            y = y.to(self.active_decoder.device).float() 
                        elif isinstance(criterion, (nn.CrossEntropyLoss)): 
                            y = y.to(self.active_decoder.device)            
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
        
        os.makedirs(f"{self.base_dir}/saved/{path}",exist_ok=True)
        if trains_decoder:
            torch.save(self.active_decoder.model.state_dict(), f"{self.base_dir}/saved/{path}/decoder.pth")
        if trains_encoder:
            torch.save(self.active_encoder.model.state_dict(), f"{self.base_dir}/saved/{path}/encoder.pth")
        if trains_adapter:
            self.model_instance.model.save_pretrained(f"{self.base_dir}/saved/{path}/adapter.pth")
        if path is not None:
            summary_metrics = self.logger.summary()
            summary_path = f"{self.base_dir}/saved/{path}/pipeline.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_metrics,f, indent=2)

    def set_eval_mode(self):
        model = self.model_instance
        # Chronos has nested `.model.model`
        if hasattr(model, "model") and hasattr(model.model, "model"):
            model.model.model.eval()
        # PaPaGei / ResNet-style (just one .model)
        elif hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.eval()
        # Direct nn.Module
        elif isinstance(model, torch.nn.Module):
            model.eval()
        else:
            return
    
    def forward(self,x,mask=None):
        if self.active_encoder is not None:
            x= self.active_encoder.forward(x)
        self.set_eval_mode()
        feats=self.model_instance.forward(x,mask)
        logits = self.active_decoder.forward((feats))
        return logits 

    def predict(self, test_loader, cfg):
        if self.active_decoder is not None:
            if hasattr(self.active_decoder,'predict'):
                with (self.logger.measure("predict", device=self.logger.device) if self.logger else nullcontext()):
                    if self.active_encoder is not None:
                        test_loader = self._encoder_loader(test_loader, cfg)
                    print("[Trainer] Extracting test embeddings...")
                    test_embed_loader = self._embed_loader(test_loader, cfg)
                    if hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model:
                        return self.active_decoder.predict(self.model_instance.model, test_embed_loader)
                    else:
                        return self.active_decoder.predict(test_embed_loader)
                        
            else:
                preds=[]
                labels=[]
                for batch in tqdm(test_loader):
                    if len(batch)==3:
                        x, mask, y = batch["x"], batch["mask"], batch["y"]
                    else:
                        x, y = batch["x"], batch["y"]
                        mask=None
                    with (self.logger.measure("predict", device=self.logger.device) if self.logger else nullcontext()):
                        logits=self.forward(x,mask)
                        if isinstance(self.active_decoder.criterion, (nn.CrossEntropyLoss)):
                            logits = torch.argmax(logits, dim=1)
                        if (hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model and hasattr(self.model_instance.model, "normalizer")):
                            logits = self.model_instance.model.normalizer(x=logits, mode="denorm")
                    preds.append(logits.detach().cpu().numpy())
                    labels.append(y.numpy())
                return np.concatenate(labels), np.concatenate(preds)
        else:
            preds,labels=self.model_instance.predict(test_loader)
            return np.concatenate(labels), np.concatenate(preds)

    def _encoder_loader(self, dataloader, cfg):
        xs=[]
        ys=[]
        for batch in dataloader:
            x,y= self.active_encoder.forward(batch)
            xs.append(x)
            ys.append(y)
        tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(x),torch.tensor(y))
        return torch.utils.data.DataLoader(tensor_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])
    
    def _embed_loader(self, dataloader, cfg):
        """
        Uses model_instance.predict() to extract embedding tensors and wraps them into a DataLoader.
        Returns: new DataLoader with (embedding, label) tensors
        """

        x, y = self.model_instance.predict(dataloader)
        tensor_dataset = torch.utils.data.TensorDataset(torch.tensor(x),torch.tensor(y))
        return torch.utils.data.DataLoader(tensor_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])
    