import pandas as pd 
import numpy as np
import os 
import sys
import joblib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
from papagei.linearprobing.utils import resample_batch_signal, load_model_without_module_prefix, get_data_for_ml
from papagei.preprocessing.ppg import preprocess_one_ppg_signal
from papagei.segmentations import waveform_to_segments, save_segments_to_directory
from sklearn.model_selection import train_test_split
from torch_ecg._preprocessors import Normalize
from papagei.models.resnet import ResNet1D, ResNet1DMoE
from papagei.linearprobing.feature_extraction_papagei import save_embeddings
from papagei.linearprobing.extracted_feature_combine import segment_avg_to_dict
from papagei.linearprobing.regression import regression_model

download_dir = "/work/pi_shenoy_umass_edu/hshastri/foundation-model-zoo/dataset/PPG-data"
df = pd.read_excel(f"{download_dir}/PPG-BP dataset.xlsx", header=1)
subjects = df.subject_ID.values
main_dir = f"{download_dir}/0_subject/"
ppg_dir = f"{download_dir}/ppg/"

if not os.path.exists(ppg_dir):
    os.mkdir(ppg_dir)
    
fs = 1000 
fs_target = 125

filenames = [f.split("_")[0] for f in os.listdir(main_dir)]

norm = Normalize(method='z-score')

for f in tqdm(filenames):
    segments = []
    for s in range(1, 4):
        print(f"Processing: {f}_{s}")
        signal = pd.read_csv(f"{main_dir}{f}_{str(s)}.txt", sep='\t', header=None)
        signal = signal.values.squeeze()[:-1]
        signal, _ = norm.apply(signal, fs=fs)
        signal, _, _, _ = preprocess_one_ppg_signal(waveform=signal,
                                                frequency=fs)
        signal = resample_batch_signal(signal, fs_original=fs, fs_target=fs_target, axis=0)
        
        padding_needed = 1250 - len(signal)
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        
        signal = np.pad(signal, pad_width=(pad_left, pad_right))
        segments.append(signal)
    segments = np.vstack(segments)
    child_dir = f.zfill(4)
    save_segments_to_directory(save_dir=ppg_dir,
                              dir_name=child_dir,
                              segments=segments)
    
df = df.rename(columns={"Sex(M/F)": "sex",
                   "Age(year)": "age",
                   "Systolic Blood Pressure(mmHg)": "sysbp",
                   "Diastolic Blood Pressure(mmHg)": "diasbp",
                   "Heart Rate(b/m)": "hr",
                   "BMI(kg/m^2)": "bmi"})
df = df.fillna(0)

# These randomly selected subject splits used in our work.
# We hardcode them because we cannot share it as a "data source".

train_ids = [  2,   6,   8,  10,  12,  15,  16,  17,  18,  19,  22,  23,  26,
        31,  32,  34,  35,  38,  40,  45,  48,  50,  53,  55,  56,  58,
        60,  61,  63,  65,  66,  83,  85,  87,  89,  92,  93,  97,  98,
        99, 100, 104, 105, 106, 107, 112, 113, 114, 116, 120, 122, 126,
       128, 131, 134, 135, 137, 138, 139, 140, 141, 146, 148, 149, 152,
       153, 154, 158, 160, 162, 164, 165, 167, 169, 170, 175, 176, 179,
       183, 184, 186, 188, 189, 190, 191, 193, 196, 197, 199, 205, 206,
       207, 209, 210, 212, 216, 217, 218, 223, 226, 227, 230, 231, 233,
       234, 240, 242, 243, 244, 246, 247, 248, 256, 257, 404, 407, 409,
       412, 414, 415, 416, 417, 419]

test_ids = [14,  21,  25,  51,  52,  62,  67,  86,  90,  96, 103, 108, 110,
       119, 123, 124, 130, 142, 144, 157, 172, 173, 174, 180, 182, 185,
       192, 195, 200, 201, 211, 214, 219, 221, 228, 239, 250, 403, 405,
       406, 410]

val_ids = [3,  11,  24,  27,  29,  30,  41,  43,  47,  64,  88,  91,  95,
       115, 125, 127, 136, 145, 155, 156, 161, 163, 166, 178, 198, 203,
       208, 213, 215, 222, 229, 232, 235, 237, 241, 245, 252, 254, 259,
       411, 418]

df_train = df[df.subject_ID.isin(train_ids)]
df_val = df[df.subject_ID.isin(val_ids)]
df_test = df[df.subject_ID.isin(test_ids)]

df_train.to_csv(f"{download_dir}/train.csv", index=False)
df_val.to_csv(f"{download_dir}/val.csv", index=False)
df_test.to_csv(f"{download_dir}/test.csv", index=False)

batch_size = 256
device = "cuda:0"
case_name = "subject_ID"
ppg_dir = f"{download_dir}/ppg/"

df_train = pd.read_csv(f"{download_dir}/train.csv")
df_val = pd.read_csv(f"{download_dir}/val.csv")
df_test = pd.read_csv(f"{download_dir}/test.csv")

df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))

dict_df = {'train': df_train, 'val': df_val, 'test': df_test}

model_config = {'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
            'n_experts': 3
            }

model = ResNet1DMoE(in_channels=1, 
            base_filters=model_config['base_filters'], 
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts'])

model_path = "../../../weights/papagei_s.pt"
model = load_model_without_module_prefix(model, model_path)
model.to(device)

# load a signal
# note that that signal is already resampled and normalized
signal = joblib.load(os.path.join(ppg_dir, '0002', '0.p'))
signal = torch.Tensor(signal)
signal = signal[None, :].unsqueeze(dim=0)
print(f"PPG dimensions before inference : {signal.shape}")

model.eval()
with torch.inference_mode():
    signal = signal.to(device)
    outputs = model(signal)
    embeddings = outputs[0].cpu().detach().numpy()
print(f"Embedding dimensions : {embeddings.shape}")

def extract_features_and_save(model, ppg_dir, batch_size, device, output_idx, resample, normalize, fs, fs_target, content):
    """
    Function to extract features and save them
    """
    for split in ['train', 'val', 'test']:
        # Choose one split at a time
        df = dict_df[split]
        save_dir = f"{download_dir}/features"

        # Creating require directory structure and names
        if not os.path.exists(f"{save_dir}"):
            os.mkdir(f"{save_dir}")
        
        model_name = model_path.split("/")[-1].split(".pt")[0]
        if not os.path.exists(f"{save_dir}/{model_name}"):
            os.mkdir(f"{save_dir}/{model_name}")
        split_dir = f"{save_dir}/{model_name}/{split}/"
        
        child_dirs = np.unique(df[case_name].values)

        # Function that extracts and saves embeddings
        save_embeddings(path=ppg_dir,
                        child_dirs=child_dirs, 
                        save_dir=split_dir, 
                        model=model, 
                        batch_size=batch_size, 
                        device=device, 
                        output_idx=output_idx,
                        resample=resample, 
                        normalize=normalize, 
                        fs=fs, 
                        fs_target=fs_target)
        
        # Compile the extracted embeddings at the patient or segment level adn save it               
        dict_feat = segment_avg_to_dict(split_dir, content)
        joblib.dump(dict_feat, f"{save_dir}/{model_name}/dict_{split}_{content}.p")

model_config = {'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
            'n_experts': 3
            }

model = ResNet1DMoE(in_channels=1, 
            base_filters=model_config['base_filters'], 
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts'])

model_path = "../../../weights/papagei_s.pt"
model = load_model_without_module_prefix(model, model_path)
model.to(device)

extract_features_and_save(model=model,
                         ppg_dir=ppg_dir,
                         batch_size=batch_size,
                         device=device,
                         output_idx=0,
                         resample=False,
                         normalize=False,
                         fs=125,
                         fs_target=125,
                         content="patient"
                         )


model_config = {'base_filters': 32,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 18,
        'n_classes': 512,
        }

model = ResNet1D(in_channels=1, 
            base_filters=model_config['base_filters'], 
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            use_mt_regression=False,
            use_projection=False)

model_path = "../../../weights/papagei_s_svri.pt"
model = load_model_without_module_prefix(model, model_path)
model.to(device)

extract_features_and_save(model=model,
                         ppg_dir=ppg_dir,
                         batch_size=batch_size,
                         device=device,
                         output_idx=0,
                         resample=False,
                         normalize=False,
                         fs=125,
                         fs_target=125,
                         content="patient"
                         )
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

save_dir = f"{download_dir}/features/"
df_train = pd.read_csv(f"{download_dir}/train.csv")
df_val = pd.read_csv(f"{download_dir}/val.csv")
df_test = pd.read_csv(f"{download_dir}/test.csv")

df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
                                                                          
case_name = "subject_ID"

def regression(save_dir, model_name, content, df_train, df_val, df_test, case_name, label):
    
    dict_train = joblib.load(f"{save_dir}/{model_name}/dict_train_{content}.p")
    dict_val = joblib.load(f"{save_dir}/{model_name}/dict_val_{content}.p")
    dict_test = joblib.load(f"{save_dir}/{model_name}/dict_test_{content}.p")
    
    X_train, y_train, _ = get_data_for_ml(df=df_train,
                                     dict_embeddings=dict_train,
                                     case_name=case_name,
                                     label=label)

    X_val, y_val, _ = get_data_for_ml(df=df_val,
                                         dict_embeddings=dict_val,
                                         case_name=case_name,
                                         label=label)
    
    X_test, y_test, _ = get_data_for_ml(df=df_test,
                                         dict_embeddings=dict_test,
                                         case_name=case_name,
                                         label=label)
    
    X_test = np.concatenate((X_test, X_val))
    y_test = np.concatenate((y_test, y_val))

    estimator = Ridge()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularization strength
        'solver': ['auto', 'cholesky', 'sparse_cg']  # Solver to use in the computational routines
    }
    
    results = regression_model(estimator=estimator,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test)
    return results


results_papagei_s_diasbp = regression(save_dir=save_dir,
                                    model_name='papagei_s',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="diasbp")

results_papagei_p_diasbp = regression(save_dir=save_dir,
                                    model_name='papagei_p',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="diasbp")

results_papagei_s_svri_diasbp = regression(save_dir=save_dir,
                                    model_name='papagei_s_svri',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="diasbp")

results_papagei_s_hr = regression(save_dir=save_dir,
                                    model_name='papagei_s',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="hr")

results_papagei_p_hr = regression(save_dir=save_dir,
                                    model_name='papagei_p',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="hr")

results_papagei_s_svri_hr = regression(save_dir=save_dir,
                                    model_name='papagei_s_svri',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="hr")


results_papagei_s_sysbp = regression(save_dir=save_dir,
                                    model_name='papagei_s',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="sysbp")

results_papagei_p_sysbp = regression(save_dir=save_dir,
                                    model_name='papagei_p',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="sysbp")

results_papagei_s_svri_sysbp = regression(save_dir=save_dir,
                                    model_name='papagei_s_svri',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="sysbp")


print(f"PaPaGei-S Diastolic BP MAE: {results_papagei_s_diasbp['mae']}")
print(f"PaPaGei-P Diastolic BP MAE: {results_papagei_p_diasbp['mae']}")
print(f"PaPaGei-S sVRI Diastolic BP MAE: {results_papagei_s_svri_diasbp['mae']}")

print(f"PaPaGei-S Systolic BP MAE: {results_papagei_s_sysbp['mae']}")
print(f"PaPaGei-P Systolic BP MAE: {results_papagei_p_sysbp['mae']}")
print(f"PaPaGei-S sVRI Systolic BP MAE: {results_papagei_s_svri_sysbp['mae']}")

print(f"PaPaGei-S HR MAE: {results_papagei_s_hr['mae']}")
print(f"PaPaGei-P HR MAE: {results_papagei_p_hr['mae']}")
print(f"PaPaGei-S sVRI HR MAE: {results_papagei_s_svri_hr['mae']}")
