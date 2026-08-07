import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import time
import os
from contextlib import nullcontext
import json
from huggingface_hub import hf_hub_download
from .metrics import get_accuracy

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
        with (self.logger.measure(f"add_decoder_{path}", device=self.logger.device) if self.logger else nullcontext()):
            if not train:
                self.decoders[decoder_name]= decoder_obj
                category = self.model_instance.model_category
                decoder_file=f"{self.base_dir}/../../models/{category}/finetuned/{path}/decoder.pth"
                if not os.path.exists(decoder_file):
                    try:
                        decoder_file = hf_hub_download(repo_id="umass-lass/fmtk-decoder-zoo",
                                                       filename=f"decoder.pth",
                                                       subfolder=f"{path}",
                                                       local_dir=f"{self.base_dir}/../../models/{category}/finetuned",
                                                       token=open(f"{self.base_dir}/../../hf-token.txt").read().strip())
                    except Exception as e:
                        raise ValueError(f"Decoder file not found at {decoder_file}")
                self.decoders[decoder_name].model.load_state_dict(torch.load(f"{decoder_file}"))
                
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

    def _checkpoint_dir(self, path):
        category = self.model_instance.model_category
        d = f"{self.base_dir}/../../models/{category}/finetuned/{path}"
        os.makedirs(d, exist_ok=True)
        return d

    def _save_checkpoint(self, path, kind, epoch, optimizer, scheduler, best_metric, run_id, trains_decoder, trains_encoder):
        state = {
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "mlflow_run_id": run_id,
        }
        if trains_decoder:
            state["decoder_state"] = self.active_decoder.model.state_dict()
        if trains_encoder:
            state["encoder_state"] = self.active_encoder.model.state_dict()
        torch.save(state, f"{self._checkpoint_dir(path)}/{kind}_checkpoint.pt")

    def _load_checkpoint(self, path, kind):
        ckpt_path = f"{self._checkpoint_dir(path)}/{kind}_checkpoint.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No {kind} checkpoint found at {ckpt_path}")
        return torch.load(ckpt_path, map_location="cpu")

    def train(self, train_loader, val_loader=None, parts_to_train=['decoder'],cfg=None,path=None,
              metric_fn=None, mlflow_cfg=None, resume_from=None):
        if val_loader is not None and metric_fn is None:
            metric_fn = get_accuracy

        trains_decoder = 'decoder' in parts_to_train
        trains_adapter = 'adapter' in parts_to_train
        trains_encoder = 'encoder' in parts_to_train

        if resume_from is not None and trains_adapter:
            raise ValueError("resume_from is not supported when 'adapter' is in parts_to_train (adapter weights aren't checkpointed).")

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

        start_epoch = 0
        best_metric = None
        resume_run_id = None
        checkpoint = None
        if resume_from is not None:
            checkpoint = self._load_checkpoint(path, resume_from)
            if trains_decoder and "decoder_state" in checkpoint:
                self.active_decoder.model.load_state_dict(checkpoint["decoder_state"])
            if trains_encoder and "encoder_state" in checkpoint:
                self.active_encoder.model.load_state_dict(checkpoint["encoder_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint.get("best_metric")
            resume_run_id = checkpoint.get("mlflow_run_id")

        use_mlflow = mlflow_cfg is not None
        if use_mlflow:
            import mlflow
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
            mlflow.set_experiment(mlflow_cfg["experiment_name"])

        log_interval = cfg.get('log_interval', 1)
        eval_metric_name = cfg.get('eval_metric', 'val_metric')

        if use_mlflow:
            mlflow_run_ctx = mlflow.start_run(run_id=resume_run_id) if resume_run_id else mlflow.start_run(run_name=mlflow_cfg.get("run_name"))
        else:
            mlflow_run_ctx = nullcontext()

        with (self.logger.measure("train", device=self.logger.device) if self.logger else nullcontext()):
            with mlflow_run_ctx:
                if use_mlflow:
                    mlflow.log_params({
                        "batch_size": cfg['batch_size'],
                        "epochs": cfg['epochs'],
                        "starting_lr": cfg['lr'],
                        "num_train_samples": len(train_loader.dataset),
                        "num_test_samples": len(val_loader.dataset) if val_loader is not None else 0,
                        "parts_to_train": ",".join(parts_to_train),
                        "eval_metric": eval_metric_name,
                        "log_interval": log_interval,
                    })
                    if mlflow_cfg.get("extra_params"):
                        mlflow.log_params(mlflow_cfg["extra_params"])

                optimizer = torch.optim.Adam(param_groups)
                criterion = getattr(self.active_decoder, "criterion")

                scheduler = None
                min_lr = cfg['lr']
                if cfg.get('scheduler'):
                    sc = cfg['scheduler']
                    if sc['type'] == 'cosine':
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sc['T_max'], eta_min=sc.get('eta_min', 0))
                        min_lr = sc.get('eta_min', 0)
                    else:
                        raise ValueError(f"Unsupported scheduler type: {sc['type']}")

                if resume_from is not None:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                        scheduler.load_state_dict(checkpoint["scheduler_state"])

                if use_mlflow:
                    mlflow.log_param("min_lr", min_lr)

                for epoch in range(start_epoch, cfg['epochs']):
                    epoch_start = time.time()
                    epoch_losses = []
                    for batch in tqdm(train_loader):
                            optimizer.zero_grad()
                            x, y = batch["x"], batch["y"]
                            mask = batch.get("mask", None)
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
                            epoch_losses.append(loss.item())

                    if scheduler is not None:
                        scheduler.step()

                    is_log_epoch = ((epoch + 1) % log_interval == 0) or (epoch == cfg['epochs'] - 1)
                    if not is_log_epoch:
                        continue

                    epoch_time = time.time() - epoch_start
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    current_lr = optimizer.param_groups[0]['lr']

                    val_metric = None
                    if val_loader is not None:
                        val_labels, val_preds = self.predict(val_loader, cfg)
                        val_metric = metric_fn(val_labels, val_preds)

                    if use_mlflow:
                        mlflow.log_metric("train_loss", avg_loss, step=epoch)
                        mlflow.log_metric("lr", current_lr, step=epoch)
                        mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)
                        if val_metric is not None:
                            mlflow.log_metric(eval_metric_name, val_metric, step=epoch)

                    current_run_id = mlflow.active_run().info.run_id if use_mlflow else None

                    compare_metric = val_metric if val_metric is not None else avg_loss
                    compare_mode = cfg.get('metric_mode', 'max') if val_metric is not None else 'min'
                    is_best = (
                        best_metric is None
                        or (compare_mode == 'max' and compare_metric > best_metric)
                        or (compare_mode == 'min' and compare_metric < best_metric)
                    )
                    if is_best:
                        best_metric = compare_metric
                        self._save_checkpoint(path, "best", epoch, optimizer, scheduler, best_metric, current_run_id, trains_decoder, trains_encoder)

                    self._save_checkpoint(path, "last", epoch, optimizer, scheduler, best_metric, current_run_id, trains_decoder, trains_encoder)

        category = self.model_instance.model_category
        os.makedirs(f"{self.base_dir}/../../models/{category}/finetuned/{path}",exist_ok=True)
        if trains_decoder:
            torch.save(self.active_decoder.model.state_dict(), f"{self.base_dir}/../../models/{category}/finetuned/{path}/decoder.pth")
        if trains_encoder:
            torch.save(self.active_encoder.model.state_dict(), f"{self.base_dir}/../../models/{category}/finetuned/{path}/encoder.pth")
        if trains_adapter:
            self.model_instance.model.save_pretrained(f"{self.base_dir}/../../models/{category}/finetuned/{path}/adapter.pth")
        if path is not None:
            summary_metrics = self.logger.summary()
            summary_path = f"{self.base_dir}/../../models/{category}/finetuned/{path}/pipeline.json"
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
                    x, y = batch["x"], batch["y"]
                    mask = batch.get("mask", None)
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
            preds, labels = [], []
            is_vlm = False
            is_generative = False
            for batch in tqdm(test_loader):
                is_vlm = 'question' in batch

                if is_vlm:
                    x = (batch['x'], batch['question'])
                    y = batch['y']
                    mask = None
                else:
                    x, y = batch['x'], batch['y']
                    mask = batch.get('mask', None)

                gpu_mem_before = self.logger.get_gpu_mem_mb() if (is_vlm and self.logger) else 0
                t0 = time.time()

                with (self.logger.measure("predict", device=self.logger.device) if self.logger else nullcontext()):
                    with torch.no_grad():
                        output = self.model_instance.forward(x, mask)

                latency_ms = (time.time() - t0) * 1000

                if is_vlm and self.logger:
                    self.logger.log_vlm_sample(
                        latency_ms=latency_ms,
                        prompt_tokens=len(batch['question'][0].split()),
                        gen_tokens=len(output[0].split()),
                        gpu_util_pct=self.logger.get_gpu_util_pct(),
                        gpu_mem_delta_mb=self.logger.get_gpu_mem_mb() - gpu_mem_before,
                    )

                # Generative output (VLM/LLM): list of strings
                if isinstance(output, list):
                    is_generative = True
                    preds.extend(output)
                    if isinstance(y, torch.Tensor):
                        labels.extend(y.tolist())
                    else:
                        labels.extend(list(y))
                else:
                    preds.append(output.detach().cpu().numpy())
                    if isinstance(y, torch.Tensor):
                        labels.append(y.numpy())
                    else:
                        labels.append(np.array(y))

            if is_generative:
                return np.array(labels, dtype=object), np.array(preds, dtype=object)
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
    