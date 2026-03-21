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
            self.dataset_instance_test = dataset_class(self.dataset_cfg, self.task_cfg, split='test')
            self.dataloader_test = DataLoader(self.dataset_instance_test, batch_size=1,
                                              shuffle=False, collate_fn=vlm_collate_fn)
        else:
            self.dataset_instance_train = dataset_class(self.dataset_cfg, self.task_cfg,split='train')
            self.dataset_instance_test = dataset_class(self.dataset_cfg, self.task_cfg,split='test')
            self.dataset_instance_val = dataset_class(self.dataset_cfg, self.task_cfg,split='val')
            self.dataloader_train = DataLoader(self.dataset_instance_train, batch_size=self.task_cfg['train_config']['batch_size'], shuffle=self.task_cfg['train_config']['shuffle'])
            self.dataloader_val = DataLoader(self.dataset_instance_val, batch_size=self.task_cfg['inference_config']['batch_size'], shuffle=self.task_cfg['inference_config']['shuffle'])
            self.dataloader_test = DataLoader(self.dataset_instance_test, batch_size=self.task_cfg['inference_config']['batch_size'], shuffle=self.task_cfg['inference_config']['shuffle'])

    def run(self):
        if self.is_vlm:
            self._run_vlm()
        else:
            self._run_tsfm()

    def _run_vlm(self):
        print(f"Running VLM inference for model: {self.model_type}_{self.model_name} on task: {self.task_name}")
        backbone_class = get_model_class(self.backbone_cfg['model_type'])
        model_hf_id = self.backbone_cfg.get('model_id', self.pipeline['backbone'])
        logger = Logger(device, f"vlm_{self.pipeline['backbone']}_{self.task_name}")

        with logger.measure("backbone", device=logger.device):
            fm_instance = backbone_class(self.device, self.model_name, self.backbone_cfg.get('model_config', {}))

        load_rec = next((r for r in logger.records if r["section"] == "backbone"), {})
        model_load_duration_sec = load_rec.get("wall_time_sec", 0)
        gpu_load_memory_mb = load_rec.get("gpu_alloc_peak", 0) / 1e6

        P = Pipeline(fm_instance, logger=logger)

        with logger.measure("inference_total", device=logger.device):
            labels_raw, preds_raw = P.predict(self.dataloader_test, cfg=self.task_cfg['inference_config'])

        # Flatten list-of-lists
        all_preds = [p for batch in preds_raw for p in batch]
        all_labels = [l for batch in labels_raw for l in batch]

        # VLM evaluation using parser + evaluator
        from fmtk.tasks.vlm_utils import get_parser, get_evaluator
        parser_fn = get_parser(self.task_cfg['parser'])
        evaluator_fn = get_evaluator(self.task_cfg['evaluator'])
        correct = sum(1 for p, g in zip(all_preds, all_labels) if evaluator_fn(parser_fn(p), g))
        total = len(all_preds)
        accuracy = correct / total if total > 0 else 0.0

        # Collect metrics — same schema as run_vlm_profile.py
        summary = logger.summary()
        vlm = summary.get('vlm', {})

        inference_rec = next(
            (r for r in logger.records if r["section"] == "inference_total"), {}
        )
        total_time = inference_rec.get("wall_time_sec", 0)
        avg_cpu_memory_usage_mb = inference_rec.get("cpu_rss_delta", 0) / 1e6

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''

        metrics = {
            "model_name":              model_hf_id.split("/")[-1],
            "dataset_name":            self.task_cfg['vlm_task_key'],
            "device":                  device,
            "model_load_duration_sec": model_load_duration_sec,
            "gpu_load_memory_mb":      gpu_load_memory_mb,
            "avg_cpu_memory_usage_mb": avg_cpu_memory_usage_mb,
            "avg_cpu_usage_percent":   0,
            "avg_gpu_usage_percent":   vlm.get("avg_gpu_util_pct", 0),
            "avg_gpu_memory_usage_mb": vlm.get("avg_gpu_mem_delta_mb", 0),
            "total_prompt_tokens":     vlm.get("total_prompt_tokens", 0),
            "total_generated_tokens":  vlm.get("total_gen_tokens", 0),
            "ttft_ms":                 0,
            "avg_latency_ms":          vlm.get("avg_latency_ms", 0),
            "throughput_tps":          vlm.get("throughput_tps", 0),
            "accuracy":                accuracy,
            "total_time":              total_time,
            "num_samples":             total,
            "gpu_name":                gpu_name,
        }

        write_header = not os.path.exists(vlm_log_file)
        with open(vlm_log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=vlm_csv_columns)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

        # Save raw Logger JSON
        logger.save()

        print(f"  Accuracy:    {correct}/{total} = {accuracy:.2%}")
        print(f"  Latency:     {metrics['avg_latency_ms']:.1f} ms/sample")
        print(f"  Throughput:  {metrics['throughput_tps']:.1f} tok/s")
        print(f"  Total time:  {total_time:.1f}s")
        print(f"  GPU util:    {metrics['avg_gpu_usage_percent']:.1f}%")

    def _run_tsfm(self):
        print(f"Running inference for model: {self.model_type}_{self.model_name} on dataset: {self.dataset_cfg['dataset_type']}")
        backbone_class = get_model_class(self.backbone_cfg['model_type'])
        logger=Logger(device,'log')
        with (logger.measure("backbone", device=logger.device) if logger else nullcontext()):
            fm_instance = backbone_class(self.device,self.model_name,self.backbone_cfg.get('model_config',{}))

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
                category = P.model_instance.model_category
                #some pipelines don't have training data times
                try:
                    with open(f"{base_dir}/../../models/{category}/finetuned/{path['path']}/pipeline.json", 'r') as file:
                        data = json.load(file)
                    summary.update({'train':data['train']})
                except Exception as e:
                    print(f"Could not load training data for {path['path']}: {e}")
                    summary.update({'train':{"gpu time": None, "gpu peak": None, "gpu energy": None}})

            if self.task_cfg['task_type']=='regression' or self.task_cfg['task_type']=='forecasting':
                    metrics = {
                            "backbone": self.pipeline['backbone'],
                            "decoder": decoders[path.get('decoder',None)]['decoder_type'],
                            "encoder": path.get('encoder',None),
                            "adapter": path.get('adapter',None),
                            "dataset_name": self.task_cfg['datasets'][0],
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                            "task_name":self.task_name ,
                            "metric":'mae',
                            "result": get_mae(y_test, y_pred),
                            "backbone memory": summary['backbone']['gpu peak'],
                            "decoder memory": summary['decoder']['gpu peak']-summary['backbone']['gpu peak'],
                            "train time":summary['train']['gpu time'],
                            "train mem peak":summary['train']['gpu peak'],
                            "train energy":summary['train']['gpu energy'],
                            "inference time":summary['predict']['gpu time'],
                            "inference mem peak":summary['predict']['gpu peak']-summary['decoder']['gpu peak'],
                            "inference energy":summary['predict']['gpu energy'],
                            }
            elif self.task_cfg['task_type']=='classification':
                metrics = {
                        "backbone": self.pipeline['backbone'],
                        "decoder": decoders[path.get('decoder',None)]['decoder_type'],
                        "encoder": path.get('encoder',None),
                        "adapter": path.get('adapter',None),
                        "dataset_name": self.task_cfg['datasets'][0],
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                        "task_name":self.task_name ,
                        "metric":'accuracy',
                        "result": get_accuracy(y_test, y_pred),
                        "backbone memory": summary['backbone']['gpu peak'],
                        "decoder memory": summary['decoder']['gpu peak']-summary['backbone']['gpu peak'],
                        "train time":summary['train']['gpu time'],
                        "train mem peak":summary['train']['gpu peak'],
                        "train energy":summary['train']['gpu energy'],
                        "inference time":summary['predict']['gpu time'],
                        "inference mem peak":summary['predict']['gpu peak']-summary['decoder']['gpu peak'],
                        "inference energy":summary['predict']['gpu energy'],
                        }

            write_header = not os.path.exists(self.log_file)
            with open(self.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics)
