import logging
from config import config
import torch
import numpy as np
from DL_Models.torch_models.torch_utils.DiceLoss import make_one_hot


def train_loop(dataloader, model, loss_name, loss_fn, optimizer):
	"""
	Performs one epoch of training the model through the dataset stored in dataloader, predicting one batch at a time 
	Using the given loss_fn and optimizer
	Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch 
	This function is called by BaseNet each epoch 
	"""

	model.train() 
	num_batches = len(dataloader)
	size = len(dataloader.dataset)

	training_loss, correct = 0, 0
	predictions, correct_pred = [], []
	for batch, (X, y) in enumerate(dataloader):
		# Move tensors to GPU
		if torch.cuda.is_available():
			X = X.cuda()
			y = y.cuda()
		# Compute prediction and loss
		pred = model(X)

		if len(y.size()) == 1:
			y = y.view(-1, 1)

		if loss_name == 'dice': # This is for segmentation task only 
			bs, sq = y.size()
			y_dice = y.long().reshape(-1, 1)
			y_dice = make_one_hot(y_dice, 3)
			y_dice = y_dice.reshape(bs * sq, -1).cuda()
			pred = pred.reshape(bs * sq, -1).cuda()
			loss = loss_fn(pred, y_dice)
		else:
			#print(pred.shape, y.shape)
			loss = loss_fn(pred, y)

		# Backpropagation and optimization 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Add up metrics
		training_loss += loss.item()
		if loss_name == 'bce':
			pred_rounded = (pred > 0.5).float()
			correct += (pred_rounded == y).float().sum()

		# Remove batch from gpu
		del X
		del y 
		torch.cuda.empty_cache()

	loss = training_loss / num_batches
	logging.info(f"Avg training loss: {loss:>7f}")
	if loss_name == 'bce':  
		accuracy = correct / size        
		logging.info(f"Avg training accuracy {accuracy:>8f}")
		return float(loss), float(accuracy) 
	return float(loss), -1 


def validation_loop(dataloader, model, loss_name, loss_fn):
	"""
	Performs one prediction through the validation set set stored in the given dataloader
	Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch 
	This function is called by BaseNet each epoch, an early stopping is implemented on the returned validation loss 
	"""

	model.eval() # eval mode for validation 
	num_batches = len(dataloader)
	size = len(dataloader.dataset)

	val_loss, correct = 0, 0
	with torch.no_grad():
		for batch, (X, y) in enumerate(dataloader):
			# Move tensors to GPU
			if torch.cuda.is_available():
				X = X.cuda()
				y = y.cuda()
			# Predict 
			pred = model(X)

			if len(y.size()) == 1:
				y = y.view(-1, 1)

			# Compute metrics
			if loss_name == 'dice': # This is for segmentation task only 
				bs, sq = y.size()
				y_dice = y.long().reshape(-1, 1)
				y_dice = make_one_hot(y_dice, 3)
				y_dice = y_dice.reshape(bs * sq, -1).cuda()
				pred = pred.reshape(bs * sq, -1).cuda()
				loss = loss_fn(pred, y_dice)
			else:
				#print(pred.shape, y.shape)
				loss = loss_fn(pred, y)

			# Add up metrics 
			val_loss += loss.item()
			if loss_name == 'bce':
				pred = (pred > 0.5).float()
				correct += (pred == y).float().sum() 

			# Remove batch from gpu
			del X
			del y 
			torch.cuda.empty_cache()

	model.train()
	
	loss = val_loss / num_batches
	logging.info(f"Avg validation loss: {loss:>8f}")
	if loss_name == 'bce':  
		accuracy = correct / size
		logging.info(f"Avg validation accuracy {accuracy:>8f}")
		return float(loss), float(accuracy)
	return float(loss), -1 # Can be used for early stopping


def test_loop(dataloader, model):
	"""
	Performs one prediction through the validation set set stored in the given dataloader
	Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch
	This function is called by BaseNet each epoch, an early stopping is implemented on the returned validation loss
	"""

	model.eval() # set nn.Module to eval mode to avoid dropout etc.
	model.cuda()

	num_batches = len(dataloader)
	size = len(dataloader.dataset)
	val_loss, correct = 0, 0
	with torch.no_grad():
		for batch, (X, _) in enumerate(dataloader):
			# Move tensors to GPU
			if torch.cuda.is_available():
				X = X.cuda()
			# Predict
			pred = model(X)
			#print(pred.shape)
			#print(pred.detach().numpy().ravel().shape)
			if batch == 0:
				all_pred = pred.cpu()
			else:
				all_pred = torch.cat((all_pred, pred.cpu()))
			del X
			torch.cuda.empty_cache()

	model.train()
	return all_pred.data.numpy()
	