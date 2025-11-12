from momentfm import MOMENTPipeline
from tqdm import tqdm
import torch
import numpy as np
from momentfm.models.statistical_classifiers import fit_svm

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'classification',
        'n_channels': 1,
        'num_class': 5
    }, # We are loading the model in `classification` mode
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)
model.init()
print(model)

from torch.utils.data import DataLoader
from momentfm.data.classification_dataset import ClassificationDataset

train_dataset = ClassificationDataset(data_split='train')
test_dataset = ClassificationDataset(data_split='test')

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

def get_embedding(model, dataloader):
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to("cuda").float()
            batch_masks = batch_masks.to("cuda")

            output = model(x_enc=batch_x, input_mask=batch_masks) # [batch_size x d_model (=1024)]
            embedding = output.embeddings
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)        

    embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
    return embeddings, labels


model.to("cuda").float()

train_embeddings, train_labels = get_embedding(model, train_dataloader)
test_embeddings, test_labels = get_embedding(model, test_dataloader)

print(train_embeddings.shape, train_labels.shape)
print(test_embeddings.shape, test_labels.shape)

clf = fit_svm(features=train_embeddings, y=train_labels)

y_pred_train = clf.predict(train_embeddings)
y_pred_test = clf.predict(test_embeddings)
train_accuracy = clf.score(train_embeddings, train_labels)
test_accuracy = clf.score(test_embeddings, test_labels)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")