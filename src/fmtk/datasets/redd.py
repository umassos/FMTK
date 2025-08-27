import numpy as np
from timeseries.datasets.base import TimeSeriesDataset
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# class REDDDataset(TimeSeriesDataset):
#     def __init__(self, dataset_cfg, task_cfg):
#         super().__init__(dataset_cfg, task_cfg)
#         # self.x_df,self.y_df = self.load_data()
#         self.scaler_x = StandardScaler()
#         self.scaler_y = StandardScaler()

#     def denormalize_y(self, y_normalized):
#         y_normalized=y_normalized.reshape(-1,1)
#         return self.scaler_y.inverse_transform(y_normalized)
    
#     def load_data(self):
#         train= self.task_cfg['train']
#         test= self.task_cfg['test']
#         base=self.dataset_cfg['dataset_path']
#         n= self.dataset_cfg['n']
#         split_factor = self.dataset_cfg['split_factor']
#         appliance=self.task_cfg['label']

#         x_train = []
#         y_train = []
#         units_to_pad = n // 2

#         #train
#         for key, values in train.items():
#             df = pd.read_csv(f"{base}/Building{key}_NILM_data_basic.csv")
#             df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
#             startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
#             endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()
            
#             if startDate > endDate:
#                 raise "Start Date must be smaller than Enddate."
            
#             df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
#             df.dropna(inplace=True)
#             x = df["main"].values
#             y = df[appliance].values
#             x = np.pad(x, (units_to_pad, units_to_pad), 'constant', constant_values = (0,0))
#             x = np.array([x[i: i + n] for i in range(len(x) - n + 1)])
#             x_train.extend(x)
#             y_train.extend(y)
        
#         x_train = np.array(x_train)  
#         y_train = np.array(y_train).reshape(-1,1)
#         x_train,x_cal , y_train, y_cal = train_test_split(x_train, y_train, test_size=split_factor, random_state=42)

#         #test
#         x_test = []
#         y_test = []
#         x_test_timestamp = []
#         for key, values in test.items():
#             df = pd.read_csv(f"{base}/Building{key}_NILM_data_basic.csv")
#             df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
#             startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
#             endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()
            
#             if startDate > endDate:
#                 raise "Start Date must be smaller than Enddate."     
#             df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
#             df.dropna(inplace=True)
#             x = df["main"].values
#             y = df[appliance].values
#             timestamp = df["Timestamp"].values
#             x = np.pad(x, (units_to_pad, units_to_pad), 'constant', constant_values = (0,0))
#             x = np.array([x[i: i + n] for i in range(len(x) - n + 1)])
#             x_test.extend(x)
#             y_test.extend(y)
#             x_test_timestamp.extend(timestamp)
    
#         x_test = np.array(x_test)
#         y_test = np.array(y_test).reshape(-1,1)
#         y_test = pd.DataFrame({self.task_cfg["label"]: y_test.ravel()})
#         y_cal=pd.DataFrame({self.task_cfg["label"]: y_cal.ravel()})

#         x_train = self.scaler_x.fit_transform(x_train)
#         y_train = self.scaler_y.fit_transform(y_train)
#         y_train = pd.DataFrame({self.task_cfg["label"]:y_train.ravel()})

#         x_cal = self.scaler_x.transform(x_cal)
#         x_test = self.scaler_x.transform(x_test)

#         x_train = np.array(x_train).reshape(x_train.shape[0], 1, n)
#         x_cal=np.array(x_cal).reshape(x_cal.shape[0], 1, n)
#         x_test = np.array(x_test).reshape(x_test.shape[0],1,n)
        

#         x_df={'train': x_train, 'val': x_cal, 'test': x_test}
#         y_df={'train': y_train, 'val': y_cal, 'test': y_test}
        
#         return x_df, y_df

#     def preprocess(self):
#         pass  # REDD-specific preprocessing

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx], np.random.randn(256)  # forecast target


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class REDDDataset(TimeSeriesDataset):
    def __init__(self, dataset_cfg, task_cfg, split="train"):
        super().__init__(dataset_cfg, task_cfg, split)
        self.task_name = self.task_cfg['task_type']
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.x_df, self.y_df = self.load_data()

    def preprocess(self):
        pass  # REDD-specific preprocessing

    def load_data(self):
        train_cfg = self.task_cfg['train']
        test_cfg = self.task_cfg['test']
        base_path = self.dataset_cfg['dataset_path']
        n = self.dataset_cfg['n']
        split_factor = self.dataset_cfg['split_factor']
        appliance = self.task_cfg['label']
        pad = n // 2

        x_train, y_train = [], []

        for key, values in train_cfg.items():
            df = pd.read_csv(f"{base_path}/Building{key}_NILM_data_basic.csv")
            df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
            df = df[
                (df["date"] >= datetime.strptime(values["start_time"], "%Y-%m-%d").date()) &
                (df["date"] <= datetime.strptime(values["end_time"], "%Y-%m-%d").date())
            ].dropna()

            x = df["main"].values
            y = df[appliance].values
            x = np.pad(x, (pad, pad), 'constant', constant_values=(0, 0))
            x = np.array([x[i:i + n] for i in range(len(x) - n + 1)])
            x_train.extend(x)
            y_train.extend(y)

        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(-1, 1)



        # Process test
        x_test, y_test = [], []

        for key, values in test_cfg.items():
            df = pd.read_csv(f"{base_path}/Building{key}_NILM_data_basic.csv")
            df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
            df = df[
                (df["date"] >= datetime.strptime(values["start_time"], "%Y-%m-%d").date()) &
                (df["date"] <= datetime.strptime(values["end_time"], "%Y-%m-%d").date())
            ].dropna()

            x = df["main"].values
            y = df[appliance].values
            x = np.pad(x, (pad, pad), 'constant', constant_values=(0, 0))
            x = np.array([x[i:i + n] for i in range(len(x) - n + 1)])
            x_test.extend(x)
            y_test.extend(y)

        x_test = np.array(x_test)
        y_test = np.array(y_test).reshape(-1, 1)

        # Normalize
        x_train = self.scaler_x.fit_transform(x_train)
        y_train = self.scaler_y.fit_transform(y_train)
        x_train, x_cal, y_train, y_cal = train_test_split(
            x_train, y_train, test_size=split_factor, random_state=42
        )
        x_test = self.scaler_x.transform(x_test)

        x_all = {
            "train": x_train,
            "val": x_cal,
            "test": x_test
        }

        y_all = {
            "train": y_train,
            "val": y_cal,
            "test": y_test
        }

        # Add channel dimension: [B, 1, L]
        x_split = np.array(x_all[self.split]).reshape(-1, 1, n)
        y_split = np.array(y_all[self.split]).reshape(-1)
        return x_split, y_split


    def __getitem__(self, idx):
        x = self.x_df[idx]  # shape: [1, n]
        y = self.y_df[idx]  # shape: scalar (float)
        return x, y

    def __len__(self):
        return len(self.x_df)

    def denormalize_y(self, y_normalized):
        y_normalized = y_normalized.reshape(-1, 1)
        y_denormalized=self.scaler_y.inverse_transform(y_normalized)
        return y_denormalized.flatten()





