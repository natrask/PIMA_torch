import os
import torch.nn as nn
import torch

def get_device(gpu):
    if gpu==-1:
        return 'cpu'
    else:
        return torch.device("cuda:{}".format(gpu))


def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def get_next_batch(dataloader):
	# Make the union of all time points and perform normalization across the whole dataset
	return dataloader.__next__()

def init_network_weights_xavier_normal(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			nn.init.constant_(m.bias, val=0)

