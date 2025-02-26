import numpy as np 
from sklearn.linear_model import LinearRegression

import torch
import torch.optim as optim 
import torch.nn as nn
from torchviz import make_dot


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

# Step 0 - Initializes parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b, w)

n_epochs = 1000

for epoch in range(n_epochs):

	# Step 1 - Computes our model's predicted output - forward pass
	yhat = b + w * x_train 


	# Step 2 - Computing the loss
	error = (yhat - y_train)
	
	loss = (error**2).mean()
	
	print(loss)
	
	
	# Step 3 - Computes gradients for both "b" and "w" parameters
	b_grad = 2 * error.mean()
	w_grad = 2 * (x_train * error).mean
	
	print(b_grad, w_grad)
	
	
	# Sets learning rate - this is "eta" ~ the "n"-like Greek letter
	lr = 0.1
	print(b, w)
	
	# Step 4 - Updates parameters using gradients and
	# the learning rate
	b = b - lr * b_grad 
	w = w - lr * w_grad  
	
	print(b, w)





