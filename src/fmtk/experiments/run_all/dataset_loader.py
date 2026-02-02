from fmtk.datasets.ppg import PPGDataset
from fmtk.datasets.redd import REDDDataset
from fmtk.datasets.ecg5000 import ECG5000Dataset
from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.datasets.uwavegesture import UWaveGestureLibraryALLDataset
from fmtk.datasets.ecl import ECLDataset
from fmtk.datasets.weather import WeatherDataset
from fmtk.datasets.traffic import TrafficDataset
from fmtk.datasets.exchange import ExchangeDataset

def get_dataset_class(dataset_type):
    if dataset_type == 'PPG-data':
        return PPGDataset
    elif dataset_type == 'REDD':
        return REDDDataset
    elif dataset_type=='ECG5000':
        return ECG5000Dataset
    elif dataset_type=='ETTh1':
        return ETTh1Dataset
    elif dataset_type=='UWaveGestureLibraryAll':
        return UWaveGestureLibraryALLDataset
    elif dataset_type=='ecl':
        return ECLDataset
    elif dataset_type=='weather':
        return WeatherDataset
    elif dataset_type=='traffic':
        return TrafficDataset
    elif dataset_type=='exchange':
        return ExchangeDataset
    elif dataset_type=='illness':
        from fmtk.datasets.illness import IllnessDataset
        return IllnessDataset
    
    raise ValueError(f"Unknown dataset type: {dataset_type}")
