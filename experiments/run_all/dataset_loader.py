def get_dataset_class(dataset_type):
    if dataset_type == 'PPG-data':
        from fmtk.datasetloaders.ppg import PPGDataset
        return PPGDataset
    elif dataset_type == 'REDD':
        from fmtk.datasetloaders.redd import REDDDataset
        return REDDDataset
    elif dataset_type=='ECG5000':
        from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
        return ECG5000Dataset
    elif dataset_type=='ETTh1':
        from fmtk.datasetloaders.etth1 import ETTh1Dataset
        return ETTh1Dataset
    elif dataset_type=='UWaveGestureLibraryAll':
        from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset
        return UWaveGestureLibraryALLDataset
    elif dataset_type=='ecl':
        from fmtk.datasetloaders.ecl import ECLDataset
        return ECLDataset
    elif dataset_type=='weather':
        from fmtk.datasetloaders.weather import WeatherDataset
        return WeatherDataset
    elif dataset_type=='traffic':
        from fmtk.datasetloaders.traffic import TrafficDataset
        return TrafficDataset
    elif dataset_type=='exchange':
        from fmtk.datasetloaders.exchange import ExchangeDataset
        return ExchangeDataset
    elif dataset_type=='illness':
        from fmtk.datasetloaders.illness import IllnessDataset
        return IllnessDataset
    
    raise ValueError(f"Unknown dataset type: {dataset_type}")
