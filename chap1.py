import numpy as np 
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim 
import torch.nn as nn
from torchviz import make_dot

# class ManualLinearRegression(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.linear = nn.Linear(1, 1)

# 	def forward(self, x):
# 		return self.linear(x)


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)


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

'''Model Training'''

n_epochs = 1000

for epoch in range(n_epochs):

	model.train()

	# Step 1 - Computes our model's predicted output - forward pass
	yhat = model(x_train_tensor)


	# Step 2 - Computing the loss
	# error = (yhat - y_train_tensor)
	# loss = (error**2).mean()
	loss = loss_fn(yhat, y_train_tensor)
	
	print("loss =\n", loss)
	print("numpy of loss =\n", loss.item())
	
	
	# Step 3 - Computes gradients for both "b" and "w" parameters
	# No more manual computation of gradients!
	# b_grad = 2 * error.mean()
	# w_grad = 2 * (x_tensor * error).mean()
	loss.backward()



	
	
	# Sets learning rate - this is "eta" ~ the "n"-like Greek letter
	print("b, w =\n", model.state_dict())
	
	# Step 4 - Updates parameters using gradients and
	# the learning rate
	# No more manual update! with torch.no_grad():
	# with torch.no_grad():
	# 	b -= lr * b.grad 
	# 	w -= lr * w.grad  
	optimizer.step()


	# b.grad.zero_(), w.grad.zero_()
	optimizer.zero_grad()

	print("b_updated, w_updated =\n", model.state_dict())





