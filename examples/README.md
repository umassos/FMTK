## ECG Classification Example with FMTK ([exp1](./examples/exp1.py))
This example demonstrates how to use FMTK for ECG signal classification using the ECG5000 dataset and Chronos foundation model. First step is to download the 
[ECG5000 dataset](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) and place it in FMTK/dataset/ECG5000 directory containing:
- ECG5000_TRAIN.ts
- ECG5000_TEST.ts

### Build and Train Pipeline

```python
chronos_logger=Logger(device,'chronos_ecg_sdk')
# Initialize pipeline with Chronos backbone
P = Pipeline(ChronosModel(device, 'tiny'),chronos_logger)

# Add SVM decoder
svm_decoder = P.add_decoder(SVMDecoder(), load=True)

# Train the decoder
P.train(dataloader_train, parts_to_train=['decoder'], cfg=train_config)

# Make predictions
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)

# Calculate accuracy
result = get_accuracy(y_test, y_pred)
```
For this example we use Chronos-tiny backbone with SVM decoder. We attach the logger to the pipeline for logging GPU memory and energy. We freeze the backbone and train decoder component and get the final prediction. 

## ETTh1 Forecasting with Papagei Example ([exp2](./examples/exp2.py))
This example demonstrates how to use FMTK for time series forecasting using the ETTh1 dataset and Papagei foundation model. First step is to download the [ETTh1 dataset](https://github.com/zhouhaoyi/ETDataset) and place it in ./dataset/ETTh1 directory containing in :
- ETTh1.csv
### Downloading Model Weights
Download PaPaGei weights as shown in [PaPaGei_github_repo](https://github.com/Nokia-Bell-Labs/papagei-foundation-model)

### Build and Train Pipeline

```python
# Papagei model configuration
model_cfg = {
    'in_channels': 1,
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,
    'n_experts': 3
}

# Initialize pipeline with Papagei backbone
P = Pipeline(PapageiModel(device, model_name='papagei_s', model_config=model_cfg))

# Add MLP decoder
mlp_decoder = P.add_decoder(MLPDecoder(device, cfg={
    'input_dim': 512,
    'output_dim': 192,
    'dropout': 0.1
}), load=True)

# Train the decoder
P.train(dataloader_train, parts_to_train=['decoder'], cfg=train_config)

# Make predictions
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)

# Calculate MAE
result = get_mae(y_test, y_pred)
```
For this example we use PaPaGei backbone with MLP decoder. The model config for PaPaGei-S model are set based on [PaPaGei github reposistry](https://github.com/Nokia-Bell-Labs/papagei-foundation-model). We freeze the backbone and train decoder component and get the final prediction. 


## PPG Regression Example with FMTK ([exp3](./examples/exp3.py))
This example demonstrates how to use FMTK for PPG-based heart rate regression using the PPG dataset and Moment foundation model with LoRA adaptation. First step is to download the [PPG dataset](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299) and place it in ./dataset/PPG-data directory.

### Build and Train Pipeline
```python
# LoRA adapter configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05
)

# Initialize pipeline with Moment backbone
P = Pipeline(MomentModel(device, 'base'))

# Add MLP decoder and channel combiner encoder
mlp_decoder = P.add_decoder(MLPDecoder(device, cfg={
    'input_dim': 1024,
    'output_dim': 1,
    'hidden_dim': 128
}, lr=train_config['lr']), load=True)
encoder = P.add_encoder(LinearChannelCombiner(num_channels=3, new_num_channels=1), load=True)

# Add LoRA adapter
peft_adapter = P.add_adapter(lora_config)

# Train all components
P.train(dataloader_train, parts_to_train=['encoder', 'decoder', 'adapter'], cfg=train_config)

# Make predictions
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)

# Calculate MAE
result = get_mae(y_test, y_pred)
```

For this example we use Moment-base backbone with LoRA adapter, linear encoder to change number of channels from 3 to 1 and MLP decoder. Based on [PEFT](https://huggingface.co/docs/peft/en/index) library we set the LORA adapter configuration. We joint train all the components and get the prediction. 