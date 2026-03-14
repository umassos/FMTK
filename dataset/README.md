# Datasets

This directory stores raw dataset files used for training and evaluation.

## Structure

```
dataset/
├── ECG5000/                # ECG classification (5000 samples)
├── ETTh1/                  # Electricity Transformer Temperature (hourly)
├── ElectricityLoad-data/   # Electricity Consuming Load (ECL)
├── Exchange/               # Exchange rate forecasting
├── PPG-data/               # Photoplethysmography (blood pressure tasks)
├── Traffic/                # Road occupancy traffic forecasting
├── UWaveGestureLibraryAll/ # Gesture classification
└── Weather/                # Weather forecasting
```

## Dataset Types

| Dataset               | Task Type    | Used For                          |
|-----------------------|--------------|-----------------------------------|
| ECG5000               | Classification | ECG arrhythmia classification   |
| PPG-data              | Regression   | Diastolic/systolic BP, heart rate |
| ETTh1                 | Forecasting  | Time series forecasting           |
| ElectricityLoad-data  | Forecasting  | Time series forecasting           |
| Exchange              | Forecasting  | Time series forecasting           |
| Traffic               | Forecasting  | Time series forecasting           |
| Weather               | Forecasting  | Time series forecasting           |
| UWaveGestureLibraryAll| Classification | Gesture recognition             |