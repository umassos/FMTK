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
    elif dataset_type=='vlm':
        from fmtk.datasetloaders.vlm_dataset import VLMDataset
        return VLMDataset
    elif dataset_type=='EuroSAT':
        from fmtk.datasetloaders.EuroSAT import EuroSATDataset
        return EuroSATDataset
    elif dataset_type=='CIFAR10':
        from fmtk.datasetloaders.cifar10 import CIFAR10Dataset
        return CIFAR10Dataset
    elif dataset_type=='ShanghaiTech':
        from fmtk.datasetloaders.ShanghaiTech import ShanghaiTechDataset
        return ShanghaiTechDataset
    elif dataset_type=='NYUDepthV2':
        from fmtk.datasetloaders.nyudepthv2 import NYUDepthV2Dataset
        return NYUDepthV2Dataset
    elif dataset_type=='VOC12':
        from fmtk.datasetloaders.voc12 import VOC12Dataset
        return VOC12Dataset
    # ── LLM (text-only) datasets ──────────────────────────────────────
    elif dataset_type == 'sst2':
        from fmtk.datasetloaders.sst2 import SST2Dataset
        return SST2Dataset
    elif dataset_type == 'ag_news':
        from fmtk.datasetloaders.ag_news import AGNewsDataset
        return AGNewsDataset
    elif dataset_type == 'conll2003':
        from fmtk.datasetloaders.conll2003 import CoNLL2003Dataset
        return CoNLL2003Dataset
    elif dataset_type == 'squad':
        from fmtk.datasetloaders.squad import SQuADDataset
        return SQuADDataset
    elif dataset_type == 'cnn_dailymail':
        from fmtk.datasetloaders.cnn_dailymail import CNNDailyMailDataset
        return CNNDailyMailDataset
    elif dataset_type == 'flores':
        from fmtk.datasetloaders.flores import FLORESDataset
        return FLORESDataset
    elif dataset_type == 'gsm8k':
        from fmtk.datasetloaders.gsm8k import GSM8KDataset
        return GSM8KDataset
    elif dataset_type == 'humaneval':
        from fmtk.datasetloaders.humaneval import HumanEvalDataset
        return HumanEvalDataset
    elif dataset_type == 'hellaswag':
        from fmtk.datasetloaders.hellaswag import HellaSwagDataset
        return HellaSwagDataset
    elif dataset_type == 'fever':
        from fmtk.datasetloaders.fever import FEVERDataset
        return FEVERDataset

    raise ValueError(f"Unknown dataset type: {dataset_type}")
