import numpy as np
from fmtk.datasets.base import TimeSeriesDataset
import pandas as pd
import os
from tqdm import tqdm
from fractions import Fraction
from math import gcd
import pyPPG.preproc as PP
from dotmap import DotMap
from scipy.signal import filtfilt, resample_poly
from torch_ecg._preprocessors import Normalize

class PPGDataset(TimeSeriesDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        self.task_name = self.task_cfg['task_type']  
        self.x_df,self.y_df = self.load_data()
        self.length = len(self.x_df)
        self.label= self.task_cfg['label']

    def load_data(self):
        # Load the dataset of ground truth from the specified path in the config
        base = self.dataset_cfg['dataset_path']
        df = pd.read_excel(f"{base}/PPG-BP dataset.xlsx", header=1)
        df = df.rename(columns={"Sex(M/F)": "sex",
                   "Age(year)": "age",
                   "Systolic Blood Pressure(mmHg)": "sysbp",
                   "Diastolic Blood Pressure(mmHg)": "diasbp",
                   "Heart Rate(b/m)": "hr",
                   "BMI(kg/m^2)": "bmi"})
        df = df.fillna(0)

        subjects = df.subject_ID.values
        train_ids = [2,   6,   8,  10,  12,  15,  16,  17,  18,  19,  22,  23,  26,
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

        df_train = df[df.subject_ID.isin(train_ids)].reset_index()
        df_val = df[df.subject_ID.isin(val_ids)].reset_index()
        df_test = df[df.subject_ID.isin(test_ids)].reset_index()

        case_name = 'subject_ID'
        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))

        # Load the dataset of ground truth signal from the specified path in the config
        main_dir = f"{base}/0_subject/"        
        fs = 1000 
        fs_target = 125
        norm = Normalize(method='z-score')

        split_map = {'train': df_train, 'val': df_val, 'test': df_test}
        df_split = split_map[self.split]
        df_split.loc[:, "subject_ID"] = df_split["subject_ID"].apply(lambda x: int(x))
        x_df=[]
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Processing {self.split}"):
            subject_id = row["subject_ID"]
            segments = []
            for s in range(1, 4):
                file_path = f"{main_dir}{subject_id}_{s}.txt"

                signal = pd.read_csv(file_path, sep='\t', header=None)
                signal = signal.values.squeeze()[:-1]
                signal, _ = norm.apply(signal, fs=fs)
                signal, _, _, _ = self.preprocess(waveform=signal, frequency=fs)
                signal = self.resample_batch_signal(signal, fs_original=fs, fs_target=fs_target, axis=0)

                padding_needed = 1250 - len(signal)
                pad_left = padding_needed // 2
                pad_right = padding_needed - pad_left
                signal = np.pad(signal, pad_width=(pad_left, pad_right))
                segments.append(signal)
            segments= np.vstack(segments)
            x_df.append(segments)  # Shape: (3, 1250)
        return x_df,split_map[self.split]

    def __getitem__(self, idx):
        x = self.x_df[idx]
        if self.task_name == 'regression':
            label_name = self.task_cfg.get("label", "sysbp")
            row = self.y_df[self.label][idx]
            return x, row

    def __len__(self):
        return self.length

    def preprocess(self,waveform,frequency,fL=0.5, fH=12, order=4,smoothing_windows={"ppg":50, "vpg":10, "apg":10, "jpg":10}):
        
        """
        Preprocessing a single PPG waveform using py PPG.
        https://pyppg.readthedocs.io/en/latest/Filters.html
        
        Args:
            waveform (numpy.array): PPG waveform for processing
            frequency (int): waveform frequency
            fL (float/int): high pass cut-off for chebyshev filter
            fH (float/int): low pass cut-off for chebyshev filter
            order (int): filter order
            smoothing_windows (dictionary): smoothing window sizes in milliseconds as dictionary
        
        Returns:
            ppg (numpy.array): filtered ppg signal
            ppg_d1 (numpy.array): first derivative of filtered ppg signal
            ppg_d2 (numpy.array): second derivative of filtered ppg signal
            ppg_d3 (numpy.array): third derivative of filtered ppg signal

        """

        prep = PP.Preprocess(fL=fL,
                        fH=fH,
                        order=order,
                        sm_wins=smoothing_windows)
        
        signal = DotMap()
        signal.v = waveform
        signal.fs = frequency
        signal.filtering = True

        ppg, ppg_d1, ppg_d2, ppg_d3 = prep.get_signals(signal)

        return ppg, ppg_d1, ppg_d2, ppg_d3


    def resample_batch_signal(self,X, fs_original, fs_target, axis=-1):
        """
        Apply resampling to a 2D array with no of segments x values

        Args:
            X (np.array): 2D segments x values array
            fs_original (int/float): Source frequency 
            fs_target (int/float): Target frequency
            axis (int): index to apply the resampling.
        
        Returns:
            X (np.array): Resampled 2D segments x values array
        """
        # Convert fs_original and fs_target to Fractions
        fs_original_frac = Fraction(fs_original).limit_denominator()
        fs_target_frac = Fraction(fs_target).limit_denominator()
        
        # Find the least common multiple of the denominators
        lcm_denominator = np.lcm(fs_original_frac.denominator, fs_target_frac.denominator)
        
        # Scale fs_original and fs_target to integers
        fs_original_scaled = fs_original_frac * lcm_denominator
        fs_target_scaled = fs_target_frac * lcm_denominator
        
        # Calculate gcd of the scaled frequencies
        gcd_value = gcd(fs_original_scaled.numerator, fs_target_scaled.numerator)
        
        # Calculate the up and down factors
        up = fs_target_scaled.numerator // gcd_value
        down = fs_original_scaled.numerator // gcd_value
        
        # Perform the resampling
        X = resample_poly(X, up, down, axis=axis)
        
        return X 