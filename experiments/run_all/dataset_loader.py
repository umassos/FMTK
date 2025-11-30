def get_dataset_class(dataset_type):
    if dataset_type == 'PPG-data':
        from fmtk.datasets.ppg import PPGDataset
        return PPGDataset
    elif dataset_type == 'REDD':
        from fmtk.datasets.redd import REDDDataset
        return REDDDataset
    elif dataset_type=='ECG5000':
        from fmtk.datasets.ecg5000 import ECG5000Dataset
        return ECG5000Dataset
    elif dataset_type=='ETTh1':
        from fmtk.datasets.etth1 import ETTh1Dataset
        return ETTh1Dataset
    elif dataset_type=='UWaveGestureLibraryAll':
        from fmtk.datasets.uwavegesture import UWaveGestureLibraryALLDataset
        return UWaveGestureLibraryALLDataset
    elif dataset_type=='ecl':
        from fmtk.datasets.ecl import ECLDataset
        return ECLDataset
    elif dataset_type=='weather':
        from fmtk.datasets.weather import WeatherDataset
        return WeatherDataset
    elif dataset_type=='traffic':
        from fmtk.datasets.traffic import TrafficDataset
        return TrafficDataset
    elif dataset_type=='exchange':
        from fmtk.datasets.exchange import ExchangeDataset
        return ExchangeDataset
    elif dataset_type=='illness':
        from fmtk.datasets.illness import IllnessDataset
        return IllnessDataset
    
    raise ValueError(f"Unknown dataset type: {dataset_type}")
