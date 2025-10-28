# %% packages
from ast import Mult ## ast = abstract syntax trees
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter

# %% 
X,y = make_multilabel_classification(n_samples=10000, n_features=10,n_classes=3, n_labels = 2)

## converting into torch
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train.shape ,y_train.shape

# %% dataset and dataloder
class MultilabelDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
    
# %%
train_dataset  = MultilabelDataset(X_torch, y_torch)
# %%
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# %%
class MultilabelNet(nn.Module):
    def __init__(self,input_dim, hidden_neurons, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim,hidden_neurons)
        self.relu1 = nn.ReLU()
        self.l2  = nn.Linear(hidden_neurons, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        # x  = self.sigmoid(x)
        return x


# %% network neurons
input_dim = train_dataset.X.shape[1]
output_dim = train_dataset.y.shape[1]
hidden_neuron = 64

# %% model initialization
model = MultilabelNet(input_dim, hidden_neuron, output_dim)


# %% if training doesn't seem right trying to change learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# %%
criterion = nn.BCEWithLogitsLoss()

# %%
losses = []
slope, bias = [],[]
EPOCHS  = 300

# %% training Loop
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # average loss over all batches
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)

# %%
sns.lineplot(x = range(EPOCHS), y = losses)
# %%


# %%

model.l1.weight.shape
# %%
type(model.l1.weight[0][0])
# %%
model.l1.bias
# %%
model.l1.bias.shape
# %%

# %% test the model
X_test_torch = torch.FloatTensor(X_test)


# %%
with torch.no_grad():
    y_test_hat = torch.nn.functional.sigmoid(model(X_test_torch)).round()
    
# convert [1, 1, 0] to string '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test]
y_test_str

most_common_cnt = Counter(y_test_str).most_common()[0][1]
print(f"Naive classifier: {most_common_cnt/len(y_test_str) * 100}%")

# %%
# %% Test accuracy
test_acc = accuracy_score(y_test, y_test_hat)
print(f"Test accuracy: {test_acc * 100}%")

# %%
torch.save(model.state_dict(),"multilabelclassification.pt")
# %%
