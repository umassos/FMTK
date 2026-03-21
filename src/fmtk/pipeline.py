import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import time
import os
from contextlib import nullcontext
import json
from fmtk.cache import EmbeddingCache
from fmtk.utils import AverageMeter

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

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
        self.embedding_cache = EmbeddingCache(cache_device='cpu', to_device=self.model_instance.device)
        self.predict_cache = EmbeddingCache(cache_device='cpu', to_device=self.model_instance.device)
    def add_adapter(self,peft_cfg, path=None):
        adapter_name=f'adapter_{self.adapter_id}'
        self.adapter_id+=1
        if self.model_instance.peft_enable:
            with (self.logger.measure("add_adapter", device=self.logger.device) if self.logger else nullcontext()):
                if path is not None:
                    self.model_instance.model.load_adapter(f"{self.base_dir}/saved/{path}/adapter.pth", adapter_name=adapter_name)
                else:
                    self.model_instance.model.add_adapter(adapter_name=adapter_name, peft_config=peft_cfg)
            return adapter_name
        else:
            self.model_instance.enable_peft(peft_cfg)
            return 'default'
    
    def set_adapter(self, adapter_name):
        pass
        if adapter_name not in self.model_instance.model.adapters:
            raise ValueError(f"adapter {adapter_name} not found. Available: {list(self.model_instance.model.adapters.keys())}")
        self.model_instance.model.set_adapter(adapter_name)
        return adapter_name
    
    def set_decoder(self, decoder_name):
        self.active_decoder = self.decoders[decoder_name]
    
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
            if path is not None:
                self.decoders[decoder_name]= decoder_obj
                if os.path.exists(path):
                    self.decoders[decoder_name].model.load_state_dict(torch.load(path))
                else:
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
                        x, y = batch["x"], batch["y"]
                        mask = batch.get("mask", None)
                        idx = batch.get("idx", None)
                        logits=self.forward(x,mask,idx)
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

    def train_eval(
        self, 
        train_loader, 
        test_loader, 
        val_loader=None, 
        parts_to_train=['decoder'], 
        train_cfg=None, 
        inference_cfg=None, 
        path=None,
        metric_fn=None, 
        mlflow_cfg=None,
    ):
        """
        Same as train(), but additionally runs predict() on test_loader after every epoch
        and prints the accuracy (or custom metric). 

        Parameters
        ----------
        train_loader : DataLoader
        test_loader : DataLoader
            Used for evaluation after each epoch.
        val_loader : DataLoader, optional
        parts_to_train : list[str]
        cfg : dict
            Must contain 'epochs', 'lr', 'batch_size', 'shuffle'.
        path : str, optional
            Save path for model artifacts.
        metric_fn : callable, optional
            metric_fn(y_true, y_pred) -> float. Defaults to accuracy
            (fraction of matching labels).
        mlflow_cfg : dict, optional
            If provided, enables MLflow logging. Supported keys:
                - experiment_name (str): MLflow experiment name (default: "default")
                - run_name (str): Name for this run (default: "default_run").
                    Also used to construct the artifact path under base_path.
                - base_path (str): Root directory for MLflow tracking
                    (default: "~/FMTK/mlflow"). A SQLite database (mlflow.db)
                    is created here. Artifacts go under <base_path>/<run_name>/.
                - extra_params (dict, optional): Additional params to log
        """
        if metric_fn is None:
            def metric_fn(y_true, y_pred):
                return (y_true == y_pred).mean()

        trains_decoder = 'decoder' in parts_to_train
        trains_adapter = 'adapter' in parts_to_train
        trains_encoder = 'encoder' in parts_to_train

        param_groups = []
        if self.active_encoder is not None:
            if trains_encoder:
                if hasattr(self.active_encoder, 'fit'):
                    train_loader = self.active_encoder.fit(train_loader)
                else:
                    encoder_params = list(self.active_encoder.trainable_parameters())
                    if len(encoder_params):
                        param_groups.append({"params": encoder_params, "lr": train_cfg['lr']})
                    if hasattr(self.active_decoder, 'fit'):
                        raise ValueError("Need differentiable decoder as attached encoder.")
            else:
                train_loader = self._encoder_loader(train_loader, train_cfg)
        if trains_adapter:
            adapter_params = list(self.model_instance.adapter_trainable_parameters())
            param_groups.append({"params": adapter_params, "lr": train_cfg['lr']})
            if hasattr(self.active_decoder, 'fit'):
                raise ValueError("Need differentiable decoder as attached adapter.")
        if trains_decoder:
            if hasattr(self.active_decoder, 'fit'):
                print("[Trainer] Extracting embeddings...")
                with (self.logger.measure("train", device=self.logger.device) if self.logger else nullcontext()):
                    train_loader = self._embed_loader(train_loader, train_cfg)
                    if hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model:
                        self.active_decoder.fit(self.model_instance.model, train_loader, train_cfg)
                        return
                    else:
                        self.active_decoder.fit(train_loader, train_cfg)
                        return
            else:
                dec_params = list(self.active_decoder.trainable_parameters())
                if len(dec_params):
                    param_groups.append({"params": dec_params, "lr": train_cfg['lr']})

        use_cache = train_cfg.get("use_cache", False)
        # --- MLflow setup ---
        mlflow_active = mlflow_cfg is not None and HAS_MLFLOW
        if mlflow_cfg is not None and not HAS_MLFLOW:
            print("[Pipeline] mlflow_cfg provided but mlflow is not installed. Skipping MLflow logging.")
        if mlflow_active:
            mlflow_base_path = os.path.expanduser(
                mlflow_cfg.get("base_path", "~/FMTK/mlflow")
            )
            run_name = mlflow_cfg.get("run_name", "default_run")
            os.makedirs(mlflow_base_path, exist_ok=True)
            db_path = os.path.join(mlflow_base_path, "mlflow.db")
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "default"))
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "epochs": train_cfg["epochs"],
                "lr": train_cfg["lr"],
                "batch_size": train_cfg.get("batch_size", 32),
                "parts_to_train": str(parts_to_train),
                "optimizer": "Adam",
            })
            if "extra_params" in mlflow_cfg:
                mlflow.log_params(mlflow_cfg["extra_params"])

        with (self.logger.measure("train", device=self.logger.device) if self.logger else nullcontext()):
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9,0.999), eps=1e-8)
            criterion = getattr(self.active_decoder, "criterion")
            train_start_time = time.time()
            time_meter = AverageMeter()
            for epoch in range(train_cfg['epochs']):
                epoch_start_time = time.time()
                loss_meter = AverageMeter()
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}"):
                    optimizer.zero_grad()
                    idx = batch.get("idx", None)
                    x, y = batch["x"], batch["y"]
                    mask = batch.get("mask", None)
                    logits = self.forward(x, mask, idx, use_cache=use_cache)
                    print(logits.shape)
                    
                    if (hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model and hasattr(self.model_instance.model, "normalizer")):
                        logits = self.model_instance.model.normalizer(x=logits, mode="denorm")
                    if isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)):
                        logits = logits.float()
                        y = y.to(self.active_decoder.device).float()
                    elif isinstance(criterion, (nn.CrossEntropyLoss)):
                        y = y.to(self.active_decoder.device)
                    elif isinstance(y, (list, tuple)) and y and isinstance(y[0], dict):
                        dev = self.active_decoder.device
                        y = [{k: v.to(dev) if isinstance(v, torch.Tensor) else v
                              for k, v in t.items()} for t in y]
                    loss = criterion(logits, y)
                    if isinstance(loss, dict):
                        loss = sum(loss.values())
                    loss_meter.update(loss.item())
                    loss.backward()
                    optimizer.step()
                epoch_end_time = time.time()
                time_meter.update(epoch_end_time - epoch_start_time)

                # --- Evaluate after this epoch ---
                y_true, y_pred = self.predict(test_loader, cfg=inference_cfg)
                score = metric_fn(y_true, y_pred)
                print(f"Epoch {epoch+1}/{train_cfg['epochs']}  loss: {loss_meter.avg:.4f}  metric: {score:.4f}")

                
                if mlflow_active:
                    mlflow.log_metrics({
                        "train_loss": loss_meter.avg,
                        "eval_metric": score,
                        "epoch_time_s": epoch_end_time - epoch_start_time,
                    }, step=epoch + 1)
            
            train_end_time = time.time()
            total_training_time = train_end_time - train_start_time
            print(f'Total training time: {total_training_time:.2f}s')
            print(f"Avg. time for {train_cfg['epochs']} epochs: {time_meter.avg:.2f}s")

            if mlflow_active:
                mlflow.log_metrics({
                    "total_training_time_s": total_training_time,
                    "avg_epoch_time_s": time_meter.avg,
                    "final_metric": score,
                    "final_loss": loss_meter.avg,
                })

        if path is not None:
            print("making dirs")
            os.makedirs(f"{self.base_dir}/saved/{path}", exist_ok=True)
            if trains_decoder:
                torch.save(self.active_decoder.model.state_dict(), f"{self.base_dir}/saved/{path}/decoder.pth")
            if trains_encoder:
                torch.save(self.active_encoder.model.state_dict(), f"{self.base_dir}/saved/{path}/encoder.pth")
            if trains_adapter:
                self.model_instance.model.save_pretrained(f"{self.base_dir}/saved/{path}/adapter.pth")
            if mlflow_active:
                mlflow.log_artifacts(f"{self.base_dir}/saved/{path}")

        if mlflow_active:
            mlflow.end_run()

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
    
    def forward(self, x, mask=None, idx=None, use_cache=False, cache_type="embedding"):
        if self.active_encoder is not None:
            x = self.active_encoder.forward(x)
        self.set_eval_mode()

        cache = self.predict_cache if cache_type == "predict" else self.embedding_cache

        if use_cache and idx is not None:
            # TODO: Add function to partially use cache
            # if only some of the idx are present
            if cache.contains(idx):  # currently all idx need to be present in the cache
                feats = cache.get(idx)
            else:
                feats = self.model_instance.forward(x, mask)
                cache.put(idx, feats)
        else:
            feats = self.model_instance.forward(x, mask)

        if self.active_decoder:
            logits = self.active_decoder.forward((feats))
        else:
            logits = feats
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
                preds = []
                labels = []
                use_predict_cache = cfg.get("use_predict_cache", True) if cfg else True
                for batch in tqdm(test_loader):
                    x = batch["x"]
                    y = batch["y"]
                    mask = batch.get("mask", None)
                    idx = batch.get("idx", None)
                    with (self.logger.measure("predict", device=self.logger.device) if self.logger else nullcontext()):
                        logits = self.forward(x, mask, idx, use_cache=use_predict_cache, cache_type="predict")
                        if isinstance(self.active_decoder.criterion, (nn.CrossEntropyLoss)):
                            logits = torch.argmax(logits, dim=1)
                        if (hasattr(self.active_decoder, "requires_model") and self.active_decoder.requires_model and hasattr(self.model_instance.model, "normalizer")):
                            logits = self.model_instance.model.normalizer(x=logits, mode="denorm")
                    if isinstance(logits, torch.Tensor):
                        preds.append(logits.detach().cpu().numpy())
                        labels.append(y.numpy())
                    else:
                        preds.append(logits)
                        labels.append(y)
                if isinstance(preds[0], np.ndarray):
                    return np.concatenate(labels), np.concatenate(preds)
                return labels, preds  # (y_true, y_pred) for detection
                
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
    