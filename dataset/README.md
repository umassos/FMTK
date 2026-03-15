# Datasets

This directory stores raw dataset files used for training and evaluation.

## Structure

```
dataset/
├── ECG5000/                    # ECG classification (5000 samples)
├── ETTh1/                      # Electricity Transformer Temperature (hourly)
├── ElectricityLoad-data/       # Electricity Consuming Load (ECL)
├── Exchange/                   # Exchange rate forecasting
├── PPG-data/                   # Photoplethysmography (blood pressure tasks)
├── Traffic/                    # Road occupancy traffic forecasting
├── UWaveGestureLibraryAll/     # Gesture classification
├── Weather/                    # Weather forecasting
└── vlm/                        # Vision-Language Model tasks
    ├── activity_recognition/
    ├── crowd_counting/
    ├── gesture_recognition/
    ├── image_classification/
    ├── object_detection/
    ├── ocr/
    ├── scene_classification/
    ├── traffic_classification/
    └── vqa/
```

## Dataset Types

### Time-Series Foundation Models (TSFM)

| Dataset                | Task Type      | Used For                          |
|------------------------|----------------|-----------------------------------|
| ECG5000                | Classification | ECG arrhythmia classification     |
| PPG-data               | Regression     | Diastolic/systolic BP, heart rate |
| ETTh1                  | Forecasting    | Time series forecasting           |
| ElectricityLoad-data   | Forecasting    | Time series forecasting           |
| Exchange               | Forecasting    | Time series forecasting           |
| Traffic                | Forecasting    | Time series forecasting           |
| Weather                | Forecasting    | Time series forecasting           |
| UWaveGestureLibraryAll | Classification | Gesture recognition               |

### Vision-Language Models (VLM)

| Dataset                    | Task Type      | Used For                        |
|----------------------------|----------------|---------------------------------|
| vlm/activity_recognition/  | Classification | Human activity recognition      |
| vlm/crowd_counting/        | Regression     | Crowd count estimation          |
| vlm/gesture_recognition/   | Classification | Hand gesture recognition        |
| vlm/image_classification/  | Classification | General image classification    |
| vlm/object_detection/      | Classification | Object category identification  |
| vlm/ocr/                   | Classification | Optical character recognition   |
| vlm/scene_classification/  | Classification | Scene/environment recognition   |
| vlm/traffic_classification/| Classification | Traffic sign/scene recognition  |
| vlm/vqa/                   | Classification | Visual question answering       |