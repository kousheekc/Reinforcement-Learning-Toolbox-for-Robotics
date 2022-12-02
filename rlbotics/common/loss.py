import torch.nn as nn


def losses(loss_name):
	if loss_name == 'mse':
		return nn.MSELoss()
	elif loss_name == 'mae':
		return nn.L1Loss()
	elif loss_name == 'cross_entropy':
		return nn.CrossEntropyLoss()
	elif loss_name == 'nnl':
		return nn.NLLLoss()
	elif loss_name == 'huber_loss':
		return nn.SmoothL1Loss()
	else:
		raise NotImplementedError
