from component_loader import get_model_class, get_decoder_class, get_encoder_class, get_adapter_class
from dataset_loader import get_dataset_class
from fmtk.metrics import get_mae, get_accuracy
from fmtk.tasks.vlm_utils import get_parser, get_evaluator
from torch.utils.data import DataLoader
from fmtk.pipeline import Pipeline
from config import *
from fmtk.utils import control_randomness
from fmtk.logger import Logger
import csv
import json
import os
import torch

class InferencePipeline:
    def __init__(self,task_name,task_info, pipeline, log_file):

        self.backbone_cfg = backbones[pipeline['backbone']]
        self.dataset_cfg = datasets[task_info['datasets'][0]]
        self.train=task_info['train']
        self.task_cfg = task_info
        self.task_name=task_name
        self.pipeline=pipeline
        self.model_name = self.backbone_cfg['model_name']
        self.model_type = self.backbone_cfg['model_type']
        self.device = device
        self.is_vlm = task_info.get('task_type') == 'vlm'
        self.log_file = log_file
        control_randomness(13)

        dataset_class = get_dataset_class(self.dataset_cfg['dataset_type'])

        if self.is_vlm:
            from fmtk.tasks.vlm_utils import TASK_REGISTRY
            vlm_task_key = self.task_cfg['vlm_task_key']
            self.task_cfg['prompt'] = TASK_REGISTRY[vlm_task_key]['prompt']

            from fmtk.datasetloaders.vlm_dataset import vlm_collate_fn
            self.dataset_instance_train = dataset_class(self.dataset_cfg, self.task_cfg, split='train')
            self.dataset_instance_test = dataset_class(self.dataset_cfg, self.task_cfg, split='test')
            self.dataloader_train = DataLoader(self.dataset_instance_train, batch_size=self.task_cfg['train_config']['batch_size'],
                                               shuffle=self.task_cfg['train_config']['shuffle'], collate_fn=vlm_collate_fn)
            self.dataloader_test = DataLoader(self.dataset_instance_test, batch_size=1,
                                              shuffle=self.task_cfg['inference_config']['shuffle'], collate_fn=vlm_collate_fn)
        else:
            self.dataset_instance_train = dataset_class(self.dataset_cfg, self.task_cfg,split='train')
            self.dataset_instance_test = dataset_class(self.dataset_cfg, self.task_cfg,split='test')
            self.dataset_instance_val = dataset_class(self.dataset_cfg, self.task_cfg,split='val')
            self.dataloader_train = DataLoader(self.dataset_instance_train, batch_size=self.task_cfg['train_config']['batch_size'], shuffle=self.task_cfg['train_config']['shuffle'])
            self.dataloader_val = DataLoader(self.dataset_instance_val, batch_size=self.task_cfg['inference_config']['batch_size'], shuffle=self.task_cfg['inference_config']['shuffle'])
            self.dataloader_test = DataLoader(self.dataset_instance_test, batch_size=self.task_cfg['inference_config']['batch_size'], shuffle=self.task_cfg['inference_config']['shuffle'])

    def run(self):
        print(f"Running {'VLM' if self.is_vlm else ''} inference for model: {self.model_type}_{self.model_name} on task: {self.task_name}")
        backbone_class = get_model_class(self.backbone_cfg['model_type'])
        model_hf_id = self.backbone_cfg.get('model_id', self.pipeline['backbone'])
        logger_name = f"vlm_{self.pipeline['backbone']}_{self.task_name}" if self.is_vlm else 'log'
        logger = Logger(device, logger_name)

        with logger.measure("backbone", device=logger.device):
            fm_instance = backbone_class(self.device, self.model_name, self.backbone_cfg.get('model_config', {}))

        P = Pipeline(fm_instance, logger=logger)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''

        for path in self.pipeline['paths']:
            if 'decoder' in path:
                decoder_class = get_decoder_class(self.task_cfg['task_type'], decoders[path['decoder']]['decoder_type'])
                with logger.measure("decoder", device=logger.device):
                    if 'decoder_config' in decoders[path['decoder']]:
                        decoder_instance = decoder_class(**decoders[path['decoder']]['decoder_config'])
                    else:
                        decoder_instance = decoder_class()
                    P.add_decoder(decoder_instance, load=True, train=self.train, path=path['path'])
            else:
                P.unload_decoder()

            if 'encoder' in path:
                encoder_class = get_encoder_class(encoders[path['encoder']]['encoder_type'])
                with logger.measure("encoder", device=logger.device):
                    if 'encoder_config' in encoders[path['encoder']]:
                        encoder_instance = encoder_class(**encoders[path['encoder']]['encoder_config'])
                    else:
                        encoder_instance = encoder_class()
                    P.add_encoder(encoder_instance, load=True)
            else:
                P.unload_encoder()

            if 'adapter' in path:
                adapter_class = get_adapter_class(adapters[path['adapter']]['adapter_type'])
                with logger.measure("adapter", device=logger.device):
                    if 'adapter_config' in adapters[path['adapter']]:
                        adapter_instance = adapter_class(**adapters[path['adapter']]['adapter_config'])
                    else:
                        adapter_instance = adapter_class()
                    P.add_adapter(adapter_instance)
            else:
                P.unload_adapter()

            if 'parts_to_train' in path and self.train:
                print(f"Training parts: {path['parts_to_train']} for path: {path['path']}")
                P.train(self.dataloader_train, parts_to_train=path['parts_to_train'],
                        cfg=self.task_cfg['train_config'], path=path['path'])
                print("Training complete")

            with logger.measure("inference_total", device=logger.device):
                labels_raw, preds_raw = P.predict(self.dataloader_test, cfg=self.task_cfg['inference_config'])

            summary = logger.summary()
            if not self.train:
                base_dir = os.path.dirname(__file__)
                category = P.model_instance.model_category
                try:
                    with open(f"{base_dir}/../../models/{category}/finetuned/{path.get('path', '')}/pipeline.json", 'r') as file:
                        data = json.load(file)
                    summary.update({'train': data['train']})
                except Exception as e:
                    # print(f"Could not load training data for {path.get('path', '')}: {e}")
                    summary.update({'train': {"gpu time": None, "gpu peak": None, "gpu energy": None}})

            if self.is_vlm:
                all_preds = [p for batch in preds_raw for p in batch]
                all_labels = [l for batch in labels_raw for l in batch]

                parser_fn = get_parser(self.task_cfg['parser'])
                evaluator_fn = get_evaluator(self.task_cfg['evaluator'])
                correct = sum(1 for p, g in zip(all_preds, all_labels) if evaluator_fn(parser_fn(p), g))
                total = len(all_preds)
                accuracy = correct / total if total > 0 else 0.0

                vlm = summary.get('vlm', {})
                total_time_s = summary.get('inference_total', {}).get('wall time', 0) / 1000
                throughput_tps = vlm.get("total_gen_tokens", 0) / total_time_s if total_time_s > 0 else 0
                metrics = {
                    # shared columns — same names as TSFM
                    "backbone":                model_hf_id.split("/")[-1],
                    "decoder":                 decoders[path['decoder']]['decoder_type'] if 'decoder' in path else None,
                    "encoder":                 path.get('encoder', None),
                    "adapter":                 path.get('adapter', None),
                    "dataset_name":            self.task_cfg['vlm_task_key'],
                    "device":                  gpu_name,
                    "task_name":               self.task_name,
                    "metric":                  "accuracy",
                    "result":                  accuracy,
                    "backbone memory(MB)":     summary['backbone']['gpu peak'],
                    "backbone load time(ms)":  summary['backbone'].get('wall time', 0),
                    "decoder memory(MB)":      summary.get('decoder', {}).get('gpu peak', None),
                    "decoder load time(ms)":   summary.get('decoder', {}).get('wall time', None),
                    "encoder memory(MB)":      summary.get('encoder', {}).get('gpu peak', None),
                    "encoder load time(ms)":   summary.get('encoder', {}).get('wall time', None),
                    "adapter memory(MB)":      summary.get('adapter', {}).get('gpu peak', None),
                    "adapter load time(ms)":   summary.get('adapter', {}).get('wall time', None),
                    "train time(ms)":         summary['train']['gpu time'],
                    "train mem peak(MB)":     summary['train']['gpu peak'],
                    "train energy(mJ)":       summary['train']['gpu energy'],
                    "inference time(ms)":     summary['predict']['wall time'],
                    "inference mem peak(MB)": summary['predict']['gpu peak'],
                    "inference energy(mJ)":   summary['predict']['gpu energy'],
                    "avg_gpu_usage_percent":   summary.get('predict', {}).get('avg gpu util pct', 0),
                    # VLM-only extras    
                    "total_prompt_tokens":     vlm.get("total_prompt_tokens", 0),
                    "total_generated_tokens":  vlm.get("total_gen_tokens", 0),
                    "throughput_tps":          throughput_tps,
                    "num_samples":             total,
                }

                write_header = not os.path.exists(vlm_log_file)
                with open(vlm_log_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=metrics.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

                logger.save()
            else:
                task_type = self.task_cfg['task_type']
                if task_type in ('regression', 'forecasting'):
                    metric_key, metric_val = 'mae', get_mae(labels_raw, preds_raw)
                else:
                    metric_key, metric_val = 'accuracy', get_accuracy(labels_raw, preds_raw)

                metrics = {
                    "backbone":           self.pipeline['backbone'],
                    "decoder":            decoders[path.get('decoder', None)]['decoder_type'],
                    "encoder":            path.get('encoder', None),
                    "adapter":            path.get('adapter', None),
                    "dataset_name":       self.task_cfg['datasets'][0],
                    "device":             gpu_name,
                    "task_name":          self.task_name,
                    "metric":             metric_key,
                    "result":             metric_val,
                    "backbone memory(MB)":    summary['backbone']['gpu peak'],
                    "backbone load time(ms)": summary['backbone'].get('wall time', 0),
                    "decoder memory(MB)":     summary['decoder']['gpu peak'],
                    "decoder load time(ms)":  summary['decoder'].get('wall time', 0),
                    "encoder memory(MB)":     summary.get('encoder', {}).get('gpu peak', None),
                    "encoder load time(ms)":  summary.get('encoder', {}).get('wall time', None),
                    "adapter memory(MB)":     summary.get('adapter', {}).get('gpu peak', None),
                    "adapter load time(ms)":  summary.get('adapter', {}).get('wall time', None),
                    "train time(ms)":         summary['train']['gpu time'],
                    "train mem peak(MB)":     summary['train']['gpu peak'],
                    "train energy(mJ)":       summary['train']['gpu energy'],
                    "inference time(ms)":     summary['predict']['wall time'],
                    "inference mem peak(MB)": summary['predict']['gpu peak'],
                    "inference energy(mJ)":   summary['predict']['gpu energy'],
                    "avg_gpu_usage_percent":  summary['predict'].get('avg gpu util pct', 0),
                }

                write_header = not os.path.exists(self.log_file)
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=metrics.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)
