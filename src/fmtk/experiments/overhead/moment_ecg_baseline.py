from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm
from torch.utils.data import DataLoader, TensorDataset
from moment.momentfm.data.classification_dataset import ClassificationDataset
from tqdm import tqdm
import numpy as np
import torch
from timeseries.logger import Logger


train_dataset = ClassificationDataset(data_split='train')
test_dataset = ClassificationDataset(data_split='test')

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

def get_embedding(model, dataloader):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(device).float()
            batch_masks = batch_masks.to(device)

            output = model(x_enc=batch_x, input_mask=batch_masks) # [batch_size x d_model (=1024)]
            embedding = output.embeddings
            # print(output.embeddings.shape)
            # embedding = output.embeddings.mean(dim=1)
            # print(embedding.shape)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)        

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels

device="cuda:0"
momentlogger=Logger(device,'moment_ecg_baseline')
with (momentlogger.measure("load backbone", device=momentlogger.device) if momentlogger else nullcontext()):
    # model = MOMENTPipeline.from_pretrained(
    #     "AutonLab/MOMENT-1-base", 
    #     model_kwargs={
    #         'task_name': 'classification',
    #         'n_channels': 1,
    #         'num_class': 5
    #     }, # We are loading the model in `classification` mode
    #     # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
    # )
    model=MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-base", 
            model_kwargs={'task_name': 'embedding','enable_gradient_checkpointing': False}, # We are loading the model in `embedding` mode to learn representations
            )
    model.init()
    model.to(device).float()

with (momentlogger.measure("train", device=momentlogger.device) if momentlogger else nullcontext()):
    train_embeddings, train_labels = get_embedding(model, train_dataloader)
    clf = fit_svm(features=train_embeddings, y=train_labels)
y_pred_train = clf.predict(train_embeddings)

for batch_x, batch_masks, batch_labels in tqdm(test_dataloader, total=len(test_dataloader)):
    single_dataset = TensorDataset(batch_x, batch_masks, batch_labels)

    # wrap it back into a DataLoader
    dataloader = DataLoader(single_dataset, batch_size=len(batch_x))
    with (momentlogger.measure("predict", device=momentlogger.device) if momentlogger else nullcontext()):
        test_embeddings, test_labels = get_embedding(model, dataloader)
        y_pred_test = clf.predict(test_embeddings)

path = momentlogger.save()
print("saved:", path)