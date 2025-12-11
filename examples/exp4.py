from fmtk.pipeline import Pipeline
from fmtk.datasets.forda import FordADataset
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.decoders.classification.svm import SVMDecoder
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader

device = 'cpu'

task_cfg = {'task_type': 'classification'}
inference_config = {'batch_size': 2, 'shuffle': False} #shuffle data
train_config = {'batch_size': 2, 'shuffle': True, 'epochs': 50, 'lr': 1e-2}
dataset_cfg = {'dataset_path': '../dataset/FordA'}

print("Starting test embedding extraction, might take a while..")

dataloader_train = DataLoader(
    FordADataset(dataset_cfg, task_cfg, split='train'),
    batch_size=train_config['batch_size'],
    shuffle=train_config['shuffle']
)

dataloader_test = DataLoader(
    FordADataset(dataset_cfg, task_cfg, split='test'),
    batch_size=inference_config['batch_size'],
    shuffle=inference_config['shuffle']
)

P = Pipeline(MomentModel(device, 'base'))
svm_decoder = P.add_decoder(SVMDecoder(), load=True)
P.train(dataloader_train, parts_to_train=['decoder'], cfg=train_config)
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
result = get_accuracy(y_test, y_pred)
print(result)
