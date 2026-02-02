from momentfm import MOMENTPipeline
import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from moment.momentfm.utils.utils import control_randomness
from moment.momentfm.data.informer_dataset import InformerDataset
from moment.momentfm.utils.forecasting_metrics import get_forecasting_metrics
from timeseries.logger import Logger

device="cuda:0"

momentlogger=Logger(device,'moment_etth1_baseline')
with (momentlogger.measure("load backbone", device=momentlogger.device) if momentlogger else nullcontext()):
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': 192,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True, # Freeze the patch embedding layer
            'freeze_embedder': True, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    )

    model.init()
    # Move the model to the GPU
    model = model.to(device)

# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=13) 

# Load data
train_dataset = InformerDataset(data_split="train", random_seed=13, forecast_horizon=192)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = InformerDataset(data_split="test", random_seed=13, forecast_horizon=192)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

criterion = torch.nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cur_epoch = 0
# max_epoch = 1

# Move the loss function to the GPU
criterion = criterion.to(device)

# # Enable mixed precision training
# scaler = torch.cuda.amp.GradScaler()

# # Create a OneCycleLR scheduler
# max_lr = 1e-4
# total_steps = len(train_loader) * max_epoch
# scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

# # Gradient clipping value
# max_norm = 5.0

with (momentlogger.measure("train", device=momentlogger.device) if momentlogger else nullcontext()):
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
        # Move the data to the GPU
        timeseries = timeseries.float().to(device)
        input_mask = input_mask.to(device)
        forecast = forecast.float().to(device)
        
        # with torch.cuda.amp.autocast():
        #     output = model(x_enc=timeseries, input_mask=input_mask)
        
        # loss = criterion(output.forecast, forecast)

        # # Scales the loss for mixed precision training
        # scaler.scale(loss).backward()

        # # Clip gradients
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad(set_to_none=True)
        output = model(x_enc=timeseries, input_mask=input_mask)
        loss = criterion(output.forecast,forecast)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    # print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

    # # Step the learning rate scheduler
    # scheduler.step()
    # cur_epoch += 1

# Evaluate the model on the test split
trues, preds, histories, losses = [], [], [], []
model.eval()
with torch.no_grad():
    for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
        with (momentlogger.measure("predict", device=momentlogger.device) if momentlogger else nullcontext()):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            # with torch.cuda.amp.autocast():
            #     output = model(x_enc=timeseries, input_mask=input_mask)
            output = model(x_enc=timeseries, input_mask=input_mask)
            loss = criterion(output.forecast, forecast)                
            losses.append(loss.item())

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.forecast.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())

trues = np.concatenate(trues, axis=0)
preds = np.concatenate(preds, axis=0)
histories = np.concatenate(histories, axis=0)

metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

# print(f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")

path = momentlogger.save()
print("saved:", path)