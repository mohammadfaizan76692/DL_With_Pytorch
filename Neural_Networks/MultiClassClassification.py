# %% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# %% data
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# %% covert to float 32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% Dataset
class IrisData(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]


# %% dataloader
iris_data = IrisData(X_train,y_train)
train_loader = DataLoader(iris_data, batch_size=32, shuffle=True)

# %% define class
class MutliClassNet(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(HIDDEN_FEATURES,NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x


# %% Hyper parameters
NUM_FEATURES  = iris_data.X.shape[1]
# print(f"NUM_FEATURES {NUM_FEATURES}")
NUM_CLASSES = len(iris_data.y.unique())
# print(f"NUM_CLASSES {NUM_CLASSES}")
HIDDEN_NEURONS = 8

# %% Model Initialization
model = MutliClassNet(NUM_FEATURES, NUM_CLASSES, HIDDEN_NEURONS)

# %% loss function optimizer
criterion = nn.CrossEntropyLoss()

# optimizer
lr = 0.01
optimizer =torch.optim.SGD(model.parameters(), lr = lr)

# %% training
EPOCHS = 500
losses  =[]
for epoch in range(1, EPOCHS+1):
    for X,y in train_loader:
        # print(X.shape)
        # print(y.shape)
        ## gradient zero
        optimizer.zero_grad()

        ## froward pass
        output = model(X)
        loss = criterion(output, y)

        ## backward pass
        loss.backward()

        ## weights update
        optimizer.step()

    losses.append(loss.item())





# %% loss over epochs
sns.lineplot(x = range(EPOCHS), y = losses)

# %% Modle Evaluation 
X_test_torch = torch.from_numpy(X_test)

with torch.no_grad():
    y_test_hat_log_softmax = model(X_test_torch)

    ## it givs maximum with the its index
    y_test_hat = torch.max(y_test_hat_log_softmax,1)

# %% looking into predictions
print(y_test_hat_log_softmax[0])
print(y_test_hat)

# %%
accuracy_score(y_test, y_test_hat.indices)

# %% Naive Base classifier , most frequent as answer
from collections import Counter
most_common_cnt = Counter(y_test).most_common()
most_common_cnt

# %%
count = most_common_cnt[0][1]
value  = most_common_cnt[0][0]
count, value
# %%
Naive_accuracy = count*100/len(y_test)
print(Naive_accuracy)
# %% saving Model
torch.save(model.state_dict(), 'multiclass_iris.pt')

# %%
