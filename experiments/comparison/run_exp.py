import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple

import aiohttp
from trace import generate_requests
from torch.utils.data import DataLoader
import requests
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.datasets.weather import WeatherDataset
from timeseries.datasets.exchange import ExchangeDataset
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from timeseries.datasets.ppg import PPGDataset
from timeseries.datasets.illness import IllnessDataset
from timeseries.datasets.ecl import ECLDataset
from timeseries.datasets.traffic import TrafficDataset
import argparse
import base64
REQUEST_LATENCY=[]

async def send_request(i, server,payload):
    st = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(server, json=payload) as resp:
                data = await resp.json()
                et = time.time()
    except Exception as e:
        et = time.time()
        print(f"Request {i} failed: {e}")  
    REQUEST_LATENCY.append((i, (et - st)))
    return (i,(et - st))

def encode_raw(arr) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),  # "float32"
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }

def to_dict(config):
    num_tasks, alpha, req_rate, cv, duration = config
    return {"num_tasks": num_tasks,
            "alpha": alpha,
            "req_rate": req_rate,
            "cv": cv,
            "duration": duration}
    
async def benchmark(
    server: str,
    input_requests: List[Tuple[str, str, int, int]],
    debug=True,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        if debug:
            print(f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                  f"{req.task}")
        await asyncio.sleep(start + req.req_time - time.time())
        batch = next(iter(req.dataloader))
        if len(batch)==3:
            payload = {
                "task": req.task,
                "x": encode_raw(batch[0].numpy()),
                "mask": encode_raw(batch[1].numpy()),
                "y": encode_raw(batch[2].numpy()),
            }
        else:
            payload = {
                "task": req.task,
                "x": encode_raw(batch[0].numpy()),
                "mask": None,
                "y": encode_raw(batch[1].numpy()),
            } 
        task = asyncio.create_task(send_request(req.req_id,server,payload))
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency


def get_res_stats(per_req_latency, benchmark_time):
    # get throughput
    per_req_latency = [i for i in per_req_latency]
    throughput = len(per_req_latency) / benchmark_time

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {throughput:.2f} requests/s")
    avg_latency = np.mean([latency for _, latency in per_req_latency])

    result = {"total_time": benchmark_time, 
              "throughput": throughput,
              "avg_latency": avg_latency}
    #create a dict from config

    res = {"config": to_dict(config), "result": result}
    
    return res


def run_exp(server, config, seed=42):
    num_tasks, alpha, req_rate, cv, duration = config

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
    
    tasks=[('hr',dataloader_hr_test),
        ('diasbp',dataloader_diasbp_test),
        ('sysbp',dataloader_sysbp_test),
        ('ecg_class',dataloader_ecg_test),
        ('gesture_class',dataloader_gesture_test),
        ('ecl',dataloader_ecl_test),
        ('traffic',dataloader_traffic_test),
        ('etth1',dataloader_etth1_test),
        ('weather',dataloader_weather_test),
        ('rate',dataloader_rate_test),
    ]
    # generate requests
    requests = generate_requests(num_tasks, alpha, req_rate, cv, duration, tasks, seed)

    # benchmark
    benchmark_start_time = time.time()
    per_req_latency = asyncio.run(benchmark(server, requests))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    res = get_res_stats(per_req_latency, benchmark_time)
    output = "result/results_client.jsonl"
    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--server", type=str, default="http://0.0.0.0:8000/predict")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    alpha,cv,duration = 1,1,120
    req_rate=6
    for i in range(5):
        num_tasks = 2
        for j in range(5):
            config = (num_tasks, alpha, req_rate, cv, duration)
            _ = run_exp(args.server, config, args.seed)
            num_tasks+=2
        req_rate+=6