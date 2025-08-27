from abc import ABC, abstractmethod

class TimeSeriesDataset(ABC):
    def __init__(self, dataset_cfg, task_cfg,split):
        self.dataset_cfg = dataset_cfg
        self.task_cfg = task_cfg
        self.split=split
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def preprocess(self):
        pass
