import numpy as np 
from sklearn.linear_model import LinearRegression 

import torch
import torch.optim as optim  
import torch.nn as nn  
from torch.utils.data import Dataset, TensorDataset, DataLoader 
from torch.utils.data.dataset import random_split 
from torch.utils.tensorboard import SummaryWriter 

import matplotlib.pyplot as plt  
# %matplotlib inline
plt.style.use('fivethirtyeight')


def make_train_step_fn(model, loss_fn, optimizer):
	def perform_train_step_fn(x, y):
		model.train()

		yhat = model(x)

		loss = loss_fn(yhat, y)
		print("loss =\n", loss)
		print("numpy of loss =\n", loss.item())

		loss.backward()

		print("b, w =\n", model.state_dict())
		optimizer.step()
		optimizer.zero_grad()
		print("b_updated, w_updated =\n", model.state_dict())

		return loss.item()

	return perform_train_step_fn

true_b = 1
true_w = 2
N = 100

np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

idx = np.arange(N)
np.random.shuffle(idx)

train_idx = idx[:int(N*.8)]
val_idx = idx[int(N*.8):]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

'''Data Preparation'''

class CustomDataset(Dataset):
	def __init__(self, x_tensor, y_tensor):
		self.x = x_tensor  
		self.y = y_tensor  

	def __getitem__(self, index):
		return (self.x[index], self.y[index])

	def __len__(self):
		return len(self.x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)

print("train_data[0]:\n", train_data[0])

'''Model Configuration'''

# Step 0 - Initializes parameters "b" and "w" randomly
torch.manual_seed(42)
# b = torch.randn(1, requires_grad = True, dtype = torch.float, device=device)
# w = torch.randn(1, requires_grad = True, dtype = torch.float, device=device)
# print(b, w)
# model = ManualLinearRegression().to(device)
model = nn.Sequential()
model.add_module('layer1', nn.Linear(1, 1))
model.to(device)

lr = 0.1

optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

'''Model Training'''

n_epochs = 1000

losses = []

for epoch in range(n_epochs):

	loss = train_step_fn(x_train_tensor, y_train_tensor)
	losses.append(loss)


