# client_queue.py
import asyncio
import aiohttp
import time
import csv
import os
import sys
from torch.utils.data import DataLoader
from timeseries.datasets.etth1 import ETTh1Dataset
from timeseries.datasets.weather import WeatherDataset
from timeseries.datasets.exchange import ExchangeDataset
from timeseries.datasets.ecg5000 import ECG5000Dataset
from timeseries.datasets.ppg import PPGDataset
from timeseries.datasets.illness import IllnessDataset
from timeseries.datasets.ecl import ECLDataset
from timeseries.datasets.traffic import TrafficDataset
import io
import base64
import numpy as np
import requests
import pyarrow as pa

SERVER = "http://0.0.0.0:8000/predict"
TASK_NAME = sys.argv[1]  # "etth1", "weather", or "rate"
RETURN_PRED = False       # include predictions in response
REQUEST_RATE = float(sys.argv[2])
interval = 1 / REQUEST_RATE

async def send_request(session, i, payload, result_time):
    st = time.time()
    try:
        async with session.post(SERVER, json=payload, timeout=300) as resp:
            data = await resp.json()
            et = time.time()
            result_time.append([
                st,                          # timestamp when request sent
                (et - st) * 1000,            # total elapsed time (ms)
                data.get("wait_time") * 1000,  # wait time in queue (ms)
                data.get("infer_time") * 1000  # GPU inference time (ms)
            ])
    except Exception as e:
        et = time.time()
        result_time.append([st, (et - st) * 1000, None, None])
        print(f"Request {i} failed: {e}")

def encode_arrow(arr) -> str:
    tensor = pa.Tensor.from_numpy(arr) 
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(tensor, sink)
    buf = sink.getvalue()
    return base64.b64encode(buf.to_pybytes()).decode("utf-8")

def encode_numpy(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def encode_raw(arr) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),  # "float32"
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }


async def run_client(dataloader, task: str):
    result_time = [["Timestamp", "TotalTime(ms)", "WaitTime(ms)", "InferTime(ms)"]]
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, batch in enumerate(dataloader):
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
            scheduled_time = start_time + i * interval
            delay = max(0, scheduled_time - time.time())
            await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send_request(session, i, payload, result_time)))

            if i == 5:  # limit for testing
                break

        await asyncio.gather(*tasks)

    return result_time

if __name__ == "__main__":
    
    inference_config = {'batch_size': 1, 'shuffle': False}
    
    if TASK_NAME == "etth1":
        dataset_cfg = {'dataset_path': '../../../dataset/ETTh1'}
        task_cfg={'task_type': 'forecasting'}
        dataloader_test = DataLoader(ETTh1Dataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "weather":
        dataset_cfg = {'dataset_path': '../../../dataset/Weather'}
        task_cfg={'task_type': 'forecasting'}
        dataloader_test = DataLoader(WeatherDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "rate":
        dataset_cfg = {'dataset_path': '../../../dataset/Exchange'}
        task_cfg={'task_type': 'forecasting'}
        dataloader_test = DataLoader(ExchangeDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "ecl":
        dataset_cfg = {'dataset_path': '../../../dataset/ECL'}
        task_cfg={'task_type': 'forecasting'}
        dataloader_test = DataLoader(ExchangeDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "illness":
        task_cfg={'task_type': 'classification'}
        dataset_cfg = {'dataset_path': '../../../dataset/illness'}
        dataloader_test = DataLoader(IllnessDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "traffic":
        task_cfg={'task_type': 'forecasting'}
        dataset_cfg = {'dataset_path': '../../../dataset/traffic'}
        dataloader_test = DataLoader(TrafficDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "ecg_class":
        task_cfg={'task_type': 'classification'}
        dataset_cfg = {'dataset_path': '../../../dataset/ECG5000'}
        dataloader_test = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "hr":
        task_cfg={'task_type': 'regression','label': 'hr'}
        dataset_cfg = {'dataset_path': '../../../dataset/PPG-data'}
        dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "diasbp":
        task_cfg={'task_type': 'regression','label': 'diasbp'}
        dataset_cfg = {'dataset_path': '../../../dataset/PPG-data'}
        dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    elif TASK_NAME == "sysbp":
        task_cfg={'task_type': 'regression','label': 'sysbp'}
        dataset_cfg = {'dataset_path': '../../../dataset/PPG-data'}
        dataloader_test = DataLoader(PPGDataset(dataset_cfg, task_cfg, split='test'),
                                     batch_size=inference_config['batch_size'], shuffle=inference_config['shuffle'])
    # Run the client
    result_time = asyncio.run(run_client(dataloader_test, TASK_NAME))

    # Save CSV
    os.makedirs(f'result/{REQUEST_RATE}', exist_ok=True)
    file_path = f'result/{REQUEST_RATE}/{TASK_NAME}.csv'
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(result_time)
    print(f"Saved results to {file_path}")
