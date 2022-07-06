import math
import torch
import torch.nn as nn
import torchvision
import logging
import pytorch_lightning as pl 
from DL_Models.torch_models.BaseNetTorch import BaseNet



class Block(nn.Module):

	def __init__(self, in_filters, out_filters, activation=nn.ReLU()):
		super(Block, self).__init__()
		#print(type(in_filters))
		self.block = nn.Sequential(
			nn.Conv1d(in_filters, out_filters, 5, stride=1, padding=2,
					  dilation=1, groups=1, bias=True,
					  padding_mode='zeros'),
			activation,
			nn.BatchNorm1d(out_filters),
			nn.Conv1d(out_filters, out_filters, 5, stride=1, padding=2,
					  dilation=1, groups=1, bias=True,
					  padding_mode='zeros'),
			activation,
			nn.BatchNorm1d(out_filters)
		)

	def forward(self, x):
		return self.block(x)


class Encoder(nn.Module):

	def __init__(self, output_size=1024, filters=(129, 128, 256, 512), pools=(4, 3, 2)):
		super(Encoder, self).__init__()
		#print(filters)
		assert ((len(filters) - 1) == len(pools))
		self.encBlocks = nn.ModuleList([Block(filters[i], filters[i + 1])
										for i in range(len(filters) - 1)])
		self.pools = [nn.MaxPool1d(pools[i]) for i in range(len(pools))]
		self.output = Block(filters[-1], output_size)

	def forward(self, x):
		output = []
		for i, block in enumerate(self.encBlocks):
			x = block(x)
			# print(f'Encoder Layer {i} shape:{x.shape}')
			output.append(x)
			x = self.pools[i](x)
			# print(f'Encoder MaxPool {i} shape:{x.shape}')
		x = self.output(x)
		# print(f'Final Encoder Layer shape:{x.shape}')
		output.append(x)
		return output


class Decoder(nn.Module):
	"""
		expects input of size (B, T, C), returns (B, T, K)
	"""

	def __init__(self, filters=(1024, 512, 256, 128), pools=(2, 2, 2)):
		super(Decoder, self).__init__()
		self.upconv = nn.ModuleList([
			nn.Sequential(
				nn.Upsample(scale_factor=pools[i], mode='linear'),
				nn.Conv1d(filters[i], filters[i + 1], kernel_size=pools[i], padding=0)
			)
			for i in range(len(pools))])
		self.decBlocks = nn.ModuleList([Block(filters[i], filters[i + 1])
										for i in range(len(filters) - 1)])

	def forward(self, x, encblocks):
		# loop through the number of channels
		# channel numbers don't match up
		for i in range(len(self.decBlocks)):
			# pass the inputs through the upsampler blocks
			x = self.upconv[i](x) # should reshape exactly such that after concatenation we again have doubling of everything
			# print(f'Upsample Layer {i} shape:{x.shape}')
			# print(f'{encblocks[i].shape}')
			encFeat = self.crop(encblocks[i], x)
			# print(f'Encfeat Layer {i} shape:{encFeat.shape}')
			x = torch.cat([x, encFeat], dim=1)
			# print(f'Crop Layer {i} shape:{x.shape}')
			x = self.decBlocks[i](x)
			# print(f'Decoder Layer {i} shape:{x.shape}')
		return x

	def crop(self, encFeatures, x):
		_, c, t = x.size()
		encFeatures = torchvision.transforms.CenterCrop([c, t])(encFeatures)
		return encFeatures



class UNet(BaseNet):
	
	def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, epochs=50, output_unit=None, encChannels=(128, 256, 512), 
			bottleneck=1024, decChannels=(512, 256, 128), pools=(3, 2, 2), retainDim=True, verbose=True, use_crf=False):
		super().__init__(model_name=model_name, path=path, model_number=model_number, loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose)
		self.timesamples = input_shape[0]
		self.input_channels = input_shape[1]
		#self.output_channels = output_shape[0]
		#self.output_width = output_shape[1]
		self.encChannels = (self.input_channels,) + encChannels
		self.decChannels = (bottleneck,) + decChannels
		self.encoder = Encoder(filters=self.encChannels, output_size=bottleneck, pools=pools)
		self.decoder = Decoder(filters=self.decChannels, pools=pools[::-1])
		#self.output_unit = get_torch_activation(output_unit) if output_unit else None 


		#self.output_layer = nn.Sequential(
		#    nn.Conv1d(in_channels=self.decChannels[-1], out_channels=self.output_channels, kernel_size=1),
		#)
		#self.out_linear = nn.Linear(in_features=self.timesamples, out_features=self.output_width)

		self.use_crf = use_crf
		if self.use_crf:
			self.crf = CRF(self.output_channels, batch_first=True)

	def get_forward_loss(self, crf):
		def loss(pred, y):
			return -self.crf.forward(pred, y, reduction='sum')
		return loss

	def forward(self, x):
		#x = x.permute(0, 2, 1)
		batchsize, channels, seq_length = x.size()
		encFeatures = self.encoder(x)
		decFeatures = self.decoder(encFeatures[::-1][0],
								   encFeatures[::-1][1:])
		#segment_map = self.output_layer(decFeatures)
		#print(segment_map.shape)
		# shape (bs, out_chan, 481)
		#_, _, segment_length = segment_map.size()
		# pad to fit
		#top = math.floor((seq_length - segment_length) / 2)
		#bottom = math.ceil((seq_length - segment_length) / 2)
		#output = nn.ZeroPad2d(padding=(bottom, top, 0, 0))(segment_map)
		#print(output.shape)
		# shape (bs, out_chan, 500)
		# Apply a linear layer before output
		#output = self.out_linear(output)
		#if self.output_width == 1: # squeeze (bs, 1, 1) to (bs, 1)
		#    output = output.squeeze(dim=2)
		#elif self.output_channels == 1: # squeee (bs, 1, 2) to (bs, 2)
		#    output = output.squeeze(dim=1)

		# Add output activation if specified (sigmoid, softmax, etc.)
		#if self.output_unit:
		#    output = self.output_unit(output)
		#print(decFeatures.shape)
		output = decFeatures

		output = torch.flatten(output, start_dim=1)
		output = self.output_layer(output)

		return output

	def predict(self, x):
		x = self.forward(x)
		if self.use_crf:
			x = self.crf.decode(x)
		return x

	def get_nb_features_output_layer(self):
		"""
		Return number of features passed into the output layer of the network 
		"""
		return 128 * 481 # TODO: remove this hard coded stuff 


if __name__ == "__main__":
	batch, chan, time = 16, 129, 500
	out_chan, out_width = 1, 1
	model = UNet(input_shape=(time, chan), output_shape=(out_chan, out_width))
	tensor = torch.randn(batch, chan, time)
	out = model(tensor)
	print(f"output shape: {out.shape}")