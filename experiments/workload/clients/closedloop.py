# client_queue.py
import requests
import time
import csv
import os
import sys
import io
import base64
import numpy as np

from torch.utils.data import DataLoader
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.datasets.weather import WeatherDataset
from timeseries.datasets.exchange import ExchangeDataset
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from timeseries.datasets.ppg import PPGDataset
from timeseries.datasets.illness import IllnessDataset
from timeseries.datasets.ecl import ECLDataset
from timeseries.datasets.traffic import TrafficDataset
from multiprocessing import Process



def send_request(i, server,payload, result_time):
    st = time.time()
    try:
        with requests.post(server, json=payload, timeout=120) as resp:
            data = resp.json()
            et = time.time()
            result_time.append([
                st,    
                data.get("arrival_time"),                      # timestamp when request sent
                (et - st) * 1000,            # total elapsed time (ms)
                data.get("wait_time") * 1000,  # wait time in queue (ms)
                data.get("decode_time") * 1000,
                data.get("infer_time") * 1000,  # GPU inference time (ms)
                data.get("server_compute_time")*1000,
                data.get("server_total_time")*1000,
                float(resp.headers.get("X-Server-Total-Time", -1))
            ])
    except Exception as e:
        et = time.time()
        result_time.append([st, (et - st) * 1000, None, None, None, None,None])
        print(f"Request {i} failed: {e}")

def encode_raw(arr) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),  # "float32"
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }


def run_client(dataloader, task: str, server: str):
    result_time = [["Timestamp", "ArrivalTime", "TotalTime(ms)", "WaitTime(ms)","DecodeTime(ms)", "InferTime(ms)", "ServerComputeTime(ms)","TotalServer(ms)","Total"]]
    start_time = time.time()
    batch = next(iter(dataloader))
    i=0
    while time.time()-start_time<=300:
        print(i)
        if len(batch)==3:
            payload = {
                "task": task,
                "x": encode_raw(batch[0].numpy()),
                "mask": encode_raw(batch[1].numpy()),
                "y": encode_raw(batch[2].numpy()),
                "return_pred": RETURN_PRED,
            }
        else:
            payload = {
                "task": task,
                "x": encode_raw(batch[0].numpy()),
                "mask": None,
                "y": encode_raw(batch[1].numpy()),
                "return_pred": RETURN_PRED,
            }              
        send_request(i,server, payload, result_time)
        i=i+1

    file_path = f'../result/{task}.csv'
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(result_time)
    print(f"Saved results to {file_path}")

if __name__ == "__main__":
    
    inference_config = {'batch_size': 1, 'shuffle': False}

    task_cfg={'task_type': 'classification'}
    dataset_cfg = {'dataset_path': '../../../../dataset/ECG5000'}
    dataloader_ecg_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg={'task_type': 'classification'}
    dataset_cfg = {'dataset_path': '../../../../dataset/UWaveGestureLibraryAll'}
    dataloader_gesture_test = DataLoader(UWaveGestureLibraryALLDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg={'task_type': 'regression','label': 'hr'}
    dataset_cfg = {'dataset_path': '../../../../dataset/PPG-data'}
    dataloader_hr_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg={'task_type': 'regression','label': 'diasbp'}
    dataset_cfg = {'dataset_path': '../../../../dataset/PPG-data'}
    dataloader_diasbp_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg={'task_type': 'regression','label': 'sysbp'}
    dataset_cfg = {'dataset_path': '../../../../dataset/PPG-data'}
    dataloader_sysbp_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg={'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': '../../../../dataset/ElectricityLoad-data'}
    dataloader_ecl_test = DataLoader(ECLDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg={'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': '../../../../dataset/Traffic'}
    dataloader_traffic_test = DataLoader(TrafficDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg={'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': '../../../../dataset/ILLNESS'}
    dataloader_illness_test = DataLoader(IllnessDataset(dataset_cfg, task_cfg, split='test',forecast_horizon=192),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg={'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': '../../../../dataset/ETTh1'}
    dataloader_etth1_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    task_cfg={'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': '../../../../dataset/Weather'}       
    dataloader_weather_test = DataLoader(WeatherDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])

    task_cfg={'task_type': 'forecasting'}
    dataset_cfg = {'dataset_path': '../../../../dataset/Exchange'}
    dataloader_rate_test = DataLoader(ExchangeDataset(dataset_cfg, task_cfg, split='test'),
                                    batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    
    tasks={
        'hr':dataloader_hr_test,
        'diasbp':dataloader_diasbp_test,
        'sysbp':dataloader_sysbp_test,
        'ecg_class':dataloader_ecg_test,
        'gesture_class':dataloader_gesture_test,
        'ecl':dataloader_ecl_test,
        'traffic':dataloader_traffic_test,
        'etth1':dataloader_etth1_test,
        'weather':dataloader_weather_test,
        'rate':dataloader_rate_test,
        }
    processes=[]
    
    servers=["http://0.0.0.0:8000/predict"]
    RETURN_PRED = False
    for i,(task_name,dataset) in enumerate(tasks.items()):
        processes.append(Process(target=run_client, args=(dataset, task_name, servers[0],)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

