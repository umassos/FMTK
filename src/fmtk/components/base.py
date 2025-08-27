from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    def preprocess(self,batch):
        pass

    def forward(self, batch):
        pass

    def postprocess(self,embedding):
        pass
    # def predict(self, dataloader):
    #     pass
    # def fit(self, dataloader, cfg):
    #     pass


