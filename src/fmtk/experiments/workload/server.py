import asyncio
import time
import threading
from typing import List, Optional, Tuple, Literal
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset
from timeseries.pipeline import Pipeline
from timeseries.components.backbones.moment import MomentModel
from timeseries.components.decoders.forecasting.mlp import MLPDecoder
from tqdm import tqdm

# -----------------------------
# Globals
# -----------------------------
SERVER = "0.0.0.0"
PORT = 8000

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_infer_lock = threading.Lock()  # still enforce single in-flight GPU inference

_pipeline = None
_etth1_decoder = None
_weather_decoder = None
_rate_decoder = None

app = FastAPI()
request_queue: asyncio.Queue = asyncio.Queue()  # async request queue

class PredictRequest(BaseModel):
    task: Literal["etth1", "weather", "rate"]
    x: List
    y: List
    return_pred: bool = False

class PredictResponse(BaseModel):
    task: str
    y_pred: Optional[List] = None
    wait_time: float  # time in queue (s)
    infer_time: float # GPU inference time (s)

def _build_pipeline_and_decoders(device: torch.device):
    P = Pipeline(MomentModel(device, "large"))

    d1 = P.add_decoder(
        MLPDecoder(device=device, cfg={"input_dim": 64*1024, "output_dim": 192, "dropout": 0.1}),
        load=True, trained=True, path="etth1_fore_momentlarge_mlp"
    )
    d2 = P.add_decoder(
        MLPDecoder(device=device, cfg={"input_dim": 64*1024, "output_dim": 192, "dropout": 0.1}),
        load=True, trained=True, path="weather_fore_momentlarge_mlp"
    )
    d3 = P.add_decoder(
        MLPDecoder(device=device, cfg={"input_dim": 64*1024, "output_dim": 192, "dropout": 0.1}),
        load=True, trained=True, path="rate_fore_momentlarge_mlp"
    )
    return P, d1, d2, d3

def _predict_one_batch(pipeline, decoder_name: str, bx: torch.Tensor, by: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dl = DataLoader(TensorDataset(bx, by), batch_size=bx.shape[0], shuffle=False)
    pipeline.load_decoder(decoder_name)
    y_test, y_pred = pipeline.predict(dl, cfg={"batch_size": bx.shape[0], "shuffle": False})
    return y_test, y_pred

@app.on_event("startup")
def _startup():
    global _pipeline, _etth1_decoder, _weather_decoder, _rate_decoder
    _pipeline, _etth1_decoder, _weather_decoder, _rate_decoder = _build_pipeline_and_decoders(_device)
    asyncio.create_task(_gpu_worker())

async def _gpu_worker():
    while True:
        fut, req, arrival_time = await request_queue.get()
        start_infer = time.time()
        with _infer_lock:
            # Convert payload to tensors
            bx = torch.tensor(req.x, dtype=torch.float32)
            by = torch.tensor(req.y, dtype=torch.float32)
            if req.task=="etth1":
                y_test, y_pred = _predict_one_batch(_pipeline, _etth1_decoder, bx, by)
            elif req.task=="weather":
                y_test, y_pred = _predict_one_batch(_pipeline, _weather_decoder, bx, by)
            elif req.task=="rate":
                y_test, y_pred = _predict_one_batch(_pipeline, _rate_decoder, bx, by)
        end_infer = time.time()
        resp = {
            "task": req.task,
            "y_pred": y_pred.detach().cpu().tolist() if req.return_pred else None,
            "wait_time": start_infer - arrival_time,
            "infer_time": end_infer - start_infer,
        }
        fut.set_result(resp)
        request_queue.task_done()

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    fut = asyncio.get_event_loop().create_future()
    arrival = time.time()
    await request_queue.put((fut, req, arrival))
    return await fut

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
