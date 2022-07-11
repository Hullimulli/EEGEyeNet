import logging
from torch import nn
import numpy as np
import torch
import math
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.parameter import Parameter
from torch.nn.modules.pooling import MaxPool1d
from abc import ABC
from DL_Models.torch_models.BaseNetTorch import BaseNet 
# import pytorch_lightning as pl


"""taken from https://github.com/wei-mao-2019/LearnTrajDep/."""

def get_dct_matrix(N):
	"""Output n*n matrix of DCT (Discrete Cosinus Transform) coefficients."""
	dct_m = np.eye(N)
	for k in np.arange(N):
		for i in np.arange(N):
			w = np.sqrt(2 / N)
			if k == 0:
				w = np.sqrt(1 / N)
			dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
	idct_m = np.linalg.inv(dct_m)
	return dct_m, idct_m


class GraphConvolution(nn.Module):
	"""adapted from https://github.com/tkipf/gcn."""

	def __init__(self, in_features, out_features, bias=True, node_n=48):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		# print(in_features)
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		self.att = Parameter(torch.FloatTensor(node_n, node_n))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):
		"""Reset parameters of GCN."""
		stdv = 1.0 / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		self.att.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		"""Forward step of GCN."""
		# print(input.shape)
		# print(self.weight.shape)
		support = torch.matmul(input, self.weight)
		output = torch.matmul(self.att, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		"""Representation function."""
		return (
			self.__class__.__name__
			+ " ("
			+ str(self.in_features)
			+ " -> "
			+ str(self.out_features)
			+ ")"
		)


class GC_Block(nn.Module):
	"""Graph convolution block."""

	def __init__(self, in_features, p_dropout, bias=True, node_n=48):
		"""Define a residual block of GCN."""
		super(GC_Block, self).__init__()
		self.in_features = in_features
		self.out_features = in_features

		self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
		self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)

		self.bn1 = nn.BatchNorm1d(node_n * in_features)
		self.bn2 = nn.BatchNorm1d(node_n * in_features)

		self.do = nn.Dropout(p_dropout)
		self.act_f = nn.Tanh()

	def forward(self, x):
		"""Forward step of GC Block."""
		y = self.gc1(x)
		b, n, f = y.shape
		y = self.bn1(y.view(b, -1)).view(b, n, f)
		y = self.act_f(y)
		y = self.do(y)

		y = self.gc2(y)
		b, n, f = y.shape
		y = self.bn2(y.view(b, -1)).view(b, n, f)
		y = self.act_f(y)
		y = self.do(y)

		return y + x

	def __repr__(self):
		"""Representation function."""
		return (
			self.__class__.__name__
			+ " ("
			+ str(self.in_features)
			+ " -> "
			+ str(self.out_features)
			+ ")"
		)


class GCN(BaseNet):
	def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
                use_residual=True, depth=12, hidden_size=64):
		"""
		We define the layers of the network in the __init__ function
		Note that this model needs an input transformation since in cannot work on sequential data
		:param input_feature: num of input feature
		:param hidden_feature: num of hidden feature
		:param p_dropout: drop out prob.
		:param num_stage: number of residual blocks
		:param node_n: number of nodes in graph
		:param dct_n: number of kept DCT coefficients
		"""
		self.timesamples = input_shape[0]
		self.input_channels = input_shape[1]
		#self.output_channels = output_shape[0]
		#self.output_width = output_shape[1]
		self.kernel_size = kernel_size 
		self.final_features=32
		self.node_n = 129
		self.num_stage = 20
		self.dct_n = self.timesamples # so that it could fit other form of data
		#self.output_unit = get_torch_activation(output_unit) if output_unit else None 

		super().__init__(model_name=model_name, path=path, model_number=model_number, loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose)

		input_feature = self.timesamples
		hidden_feature = hidden_size
		p_dropout = 0.7
		dct_matrices = get_dct_matrix(input_feature)

		if torch.cuda.is_available():
			self.dct_matrix = torch.from_numpy(dct_matrices[0]).type(torch.float32).cuda()
		else:
			self.dct_matrix = torch.from_numpy(dct_matrices[0]).type(torch.float32)

		self.sample_encoder = nn.GRU(129,64,bidirectional= True,batch_first=True,dropout = 0.5, num_layers=3)

		self.gc1 = GraphConvolution(self.dct_n, hidden_feature, node_n=self.node_n)
		self.bn1 = nn.BatchNorm1d(self.node_n * hidden_feature)

		self.gcbs = []
		for i in range(self.num_stage):
			self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=self.node_n))

		self.gcbs = nn.ModuleList(self.gcbs)

		self.gc7 = GraphConvolution(hidden_feature, self.final_features, node_n=self.node_n)

		self.do = nn.Dropout(p_dropout)
		self.act_f = nn.Tanh()

		#self.output_layer = nn.Sequential(
		#	Pad_Conv(kernel_size=self.kernel_size, value=0),
		#	nn.Conv1d(in_channels=self.nb_features, out_channels=self.output_channels, kernel_size=self.kernel_size),
			# shape (bs, output_channels, timesamples)
			#nn.Linear(in_features=self.timesamples, out_features=self.output_width)
			# shape (bs, output_channels, output_width)
		#)

	def forward(self, x):
		# Getting DCT coefficients of input
		# print(self.dct_matrix[: self.dct_n, :].shape)
		y = torch.matmul(x,self.dct_matrix[: self.dct_n, :])
		# print(x.shape)
		y = self.gc1(y)
		b, n, f = y.shape

		y = self.bn1(y.view(b, -1)).view(b, n, f)
		y = self.act_f(y)
		y = self.do(y)

		for i in range(self.num_stage):
			y = self.gcbs[i](y)

		y = self.gc7(y)
		#print(y.shape)
		# shape (bs, input_channels, final_features)

		y = torch.flatten(y, start_dim=1)
		#print(y.shape)
		# shape (bs, self.nb_features_output_channel())

		output = self.output_layer(y)
		#print(output.shape)
		# shape (bs, out_chan, out_width)

		# if 1d-target, squeeze (bs, 1, 1) to (bs, 1)
		#if self.output_width == 1:
			#output = output.squeeze(dim=2)
		#elif self.output_channels == 1:
			#output = output.squeeze(dim=1)

		# Add output activation if specified (sigmoid, softmax, etc.)
		#if self.output_unit:
			#output = self.output_unit(output)


		return output

	def get_nb_features_output_layer(self):
		"""
		Return number of features passed into the output layer of the network 
		"""
		return self.final_features*self.input_channels # + 128 for LSTM


if __name__ == "__main__":
	batch, chan, time = 16, 129, 500
	out_chan, out_width = 1,1
	model = GCNModel(input_shape=(time, chan), output_shape=(out_chan, out_width))
	tensor = torch.randn(batch, chan, time)
	out = model(tensor)
	print(f"output shape: {out.shape}")
