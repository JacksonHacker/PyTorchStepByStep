import numpy as np   

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F 
from torch.utils.data import DataLoader, TensorDataset 

from sklearn.datasets import make_moons 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, roc_curve, \
precision_recall_curve, auc  

from stepbystep.v0 import StepByStep

import matplotlib.pyplot as plt 


##################################
'''Data Generation'''

# 这是scikit-learn中的一个函数，用于生成一个二维的二分类数据集，
# 数据集中的样本点形成两个交错的半圆形（类似于月亮的形状）。
X, y = make_moons(n_samples=100, noise=0.3, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(
	X,
	y,
	test_size=.2,
	random_state=13
)


# 对训练集和验证集的特征进行标准化处理，使得数据的均值为0，方差为1。
sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

##################################
'''Data Preparation'''

torch.manual_seed(13)

x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(
	dataset=train_dataset,
	batch_size=16,
	shuffle=True
)

val_loader = DataLoader(dataset=val_dataset, batch_size=16)

##################################
'''Model Configuration'''

lr = 0.1 

torch.manual_seed(42)

model = nn.Sequential()
model.add_module('linear', nn.Linear(2, 1))

optimizer = optim.SGD(model.parameters(), lr=lr)

# Binary Cross-Entropy Loss
loss_fn = nn.BCEWithLogitsLoss()

##################################
'''Model Training'''

n_epoches = 100 

sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.train(n_epoches)

fig = sbs.plot_losses()
plt.show()

##################################
'''Prediction'''

predictions = sbs.predict(X_train[:4])
print("predictions: \n", predictions)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

probabilities = sigmoid(predictions)
print("probabilities: \n", probabilities)

classes = (predictions >= 0).astype(int)
print("classes: \n", classes)




