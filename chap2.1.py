import numpy as np 
import datetime 

import torch
import torch.optim as optim  
import torch.nn as nn  
import torch.functional as F  
from torch.utils.data import DataLoader, TensorDataset, random_split 
from torch.utils.tensorboard import SummaryWriter 

import matplotlib.pyplot as plt  
plt.style.use('fivethirtyeight')

class StepByStep(object):
	def __init__(self, model, loss_fn, optimizer):
		self.model = model
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# Let's send the model to the specified device right away
		self.model.to(self.device)

		self.train_loader = None
		self.val_loader = None 
		self.writer = None 

		# These attributes are going to be computed internally
		self.losses = []
		self.val_losses = []
		self.total_epochs = 0


		# train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
		# val_step_fn = make_val_step_fn(model, loss_fn)

		# Note: there are NO ARGS there! It makes use of the class
		# attributes directly
		self.train_step_fn = self._make_train_step_fn()
		self.val_step_fn = self._make_val_step_fn()

	def to(self, device):
		# This method allows the user to specify a different device
		try:
			self.device = device 
			self.model.to(device)
		except RuntimeError:
			self.device = ('cuda' if torch.cuda.is_available() 
							else 'cpu')
			print(f"Couldn't send it to {device}, \
				sending it to {self.device} instead.")
			self.model.to(self.device)

	def set_loaders(self, train_loader, val_loader=None):
		self.train_loader = train_loader
		self.val_loader = val_loader

	# writer = SummaryWriter('runs/simple_linear_regression')
	def set_tensorboard(self, name, folder='runs'):
		suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
		self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')


	# def make_train_step_fn(model, loss_fn, optimizer):
	# 	def perform_train_step_fn(x, y):
	# 		model.train()

	# 		yhat = model(x)

	# 		loss = loss_fn(yhat, y)
	# 		print("loss =\n", loss)
	# 		print("numpy of loss =\n", loss.item())

	# 		loss.backward()

	# 		print("b, w =\n", model.state_dict())
	# 		optimizer.step()
	# 		optimizer.zero_grad()
	# 		print("b_updated, w_updated =\n", model.state_dict())

	# 		return loss.item()

	# 	return perform_train_step_fn
	def _make_train_step_fn(self):
		def perform_train_step_fn(x, y):
			self.model.train()

			yhat = self.model(x)

			loss = self.loss_fn(yhat, y)

			loss.backward()

			self.optimizer.step()
			self.optimizer.zero_grad()

			return loss.item()

		return perform_train_step_fn

	# def make_val_step_fn(model, loss_fn):
	# 	def perform_val_step_fn(x, y):
	# 		model.eval()

	# 		yhat = model(x)

	# 		loss = loss_fn(yhat, y)

	# 		return loss.item()
	#	return perform_val_step_fn
	def _make_val_step_fn(self):
		def perform_val_step_fn(x, y):
			self.model.eval() 

			yhat = self.model(x)

			loss = loss_fn(yhat, y)

			return loss.item()

		return perform_val_step_fn


	# def mini_batch(device, data_loader, step_fn):
	# 	mini_batch_losses = []

	# 	for x_batch, y_batch in data_loader:

	# 		x_batch = x_batch.to(device)
	# 		y_batch = y_batch.to(device)

	# 		mini_batch_loss = step_fn(x_batch, y_batch)
	# 		mini_batch_losses.append(mini_batch_loss)
	
	# 	loss = np.mean(mini_batch_losses)

	# 	return loss 
	def _mini_batch(self, validation=False):
		if validation:
			data_loader = self.val_loader
			step_fn = self.val_step_fn
		else:
			data_loader = self.train_loader
			step_fn = self.train_step_fn

		if data_loader is None:
			return None

		mini_batch_losses = []

		for x_batch, y_batch in data_loader:
			x_batch = x_batch.to(self.device)
			y_batch = y_batch.to(self.device)

			mini_batch_loss = step_fn(x_batch, y_batch)
			mini_batch_losses.append(mini_batch_loss)

		loss = np.mean(mini_batch_losses)

		return loss

	def set_seed(self, seed=42):
		torch.backends.cudnn.deterministic = True 
		torch.backends.cudnn.benchmark = False 
		torch.manual_seed(seed)
		np.random.seed(seed)

	def train(self, n_epochs, seed=42):
		self.set_seed(seed)

		for epoch in range(n_epochs):
			self.total_epochs += 1

			# loss = mini_batch(device, train_loader, train_step_fn)
			# losses.append(loss)

			# # No gradients in validation!
			# with torch.no_grad():
			# 	val_loss = mini_batch(device, val_loader, val_step_fn)
			# 	val_losses.append(val_loss)

			loss = self._mini_batch(validation=False)
			self.losses.append(loss) 

			with torch.no_grad():
				val_loss = self._mini_batch(validation=True)
				self.val_losses.append(val_loss)


			# writer.add_scalars(main_tag='loss',
			# 					tag_scalar_dict={
			# 		   			'training': loss,
			# 		   			'validation': val_loss},
			# 		   			global_step=epoch)
			if self.writer:
				scalars = {'training': loss}
				if val_loss is not None:
					scalars.update({'validation': val_loss})

				self.writer.add_scalars(main_tag='loss', 
										tag_scalar_dict = scalars,
										global_step=epoch)
		# writer.close()
		if self.writer:
			self.writer.flush()

	def save_checkpoint(self, filename):
		checkpoint = {
			'epoch': self.total_epochs,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss': self.losses,
			'val_loss': self.val_losses
		}

		torch.save(checkpoint, filename)

	def load_checkpoint(self, filename):
		checkpoint = torch.load(filename)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.total_epochs = checkpoint['epoch']
		self.losses = checkpoint['loss']
		self.val_losses = checkpoint['val_loss']
		self.model.train() # always use TRAIN for resuming training

	def predict(self, x):
		self.model.eval()
		x_tensor = torch.as_tensor(x).float()
		y_hat_tensor = self.model(x_tensor.to(self.device))
		self.model.train()
		return y_hat_tensor.detach().cpu().numpy()


	def plot_losses(self):
		fig = plt.figure(figsize=(10, 4))
		plt.plot(self.losses, label='Training Loss', c='b')
		if self.val_loader:
			plt.plot(self.val_losses, label='Validation Loss', c='r')
		plt.yscale('log')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.tight_layout()
		return fig 

	# x_dummy, y_dummy = next(iter(train_loader))
	# writer.add_graph(model, x_dummy.to(device))
	def add_graph(self):
		if self.train_loader and self.writer:
			# Fetches a single mini-batch so we can use add_graph
			x_dummy, y_dummy = next(iter(self.train_loader))
			self.writer.add_graph(self.model, x_dummy.to(self.device))


#############################################
'''Data Generation'''
# data_generation/simple_linear_regression.py

true_b = 1
true_w = 2
N = 100

np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon


#############################################
'''Data Preparation'''
# %load data_preparation/v2.py

# 1. 控制数据集划分的随机性
# 2. 控制数据加载时的乱序行为
torch.manual_seed(13)

x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()

# Builds dataset containing ALL data points 
dataset = TensorDataset(x_tensor, y_tensor)

# Perform the split 
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
	dataset=train_data, 
	batch_size=16, 
	shuffle=True)

val_loader = DataLoader(
	dataset=val_data,
	batch_size=16)

#############################################
'''Model Configuration'''
# writefile model_configuration/v4.py

torch.manual_seed(42)
# b = torch.randn(1, requires_grad = True, dtype = torch.float, device=device)
# w = torch.randn(1, requires_grad = True, dtype = torch.float, device=device)
# print(b, w)
# model = ManualLinearRegression().to(device)
model = nn.Sequential()
model.add_module('layer1', nn.Linear(1, 1))


lr = 0.1

optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

# train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
# val_step_fn = make_val_step_fn(model, loss_fn)

# # Creates a Summary Writer to interface with TensorBoard
# writer = SummaryWriter('runs/simple_linear_regression')
# # Fetches a single mini-batch so we can use add_graph
# x_dummy, y_dummy = next(iter(train_loader))
# writer.add_graph(model, x_dummy.to(device))

#############################################
'''Model Training'''

sbs = StepByStep(model, loss_fn, optimizer) # From Model Configuration
sbs.set_loaders(train_loader, val_loader) # From Data Prepation
sbs.set_tensorboard('classy')

sbs.train(n_epochs=200)

print("model.state_dict(): ", model.state_dict())
print("total_epochs: ", sbs.total_epochs)

fig = sbs.plot_losses()
plt.show()


#############################################
'''Making Predictions'''

new_data = np.array([.5, .3, .7]).reshape(-1, 1)
predictions = sbs.predict(new_data)


