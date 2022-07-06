import logging
from torch import nn
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.pooling import MaxPool1d
from abc import ABC
from DL_Models.torch_models.BaseNetTorch import BaseNet

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import pytorch_lightning as pl 


class TransformerSimple(BaseNet):

    def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
                use_residual=True, depth=12, hidden_size=129, dropout=0.5):
        """
        We define the layers of the network in the __init__ function
        """
        self.timesamples = input_shape[0]
        self.input_channels = input_shape[1]
        #self.output_channels = output_shape[0]
        #self.output_width = output_shape[1]
        super().__init__(model_name=model_name, path=path, model_number=model_number, loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose)
       
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_channels, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        #self.reduction_head = ReductionHead(129,1000)
        #self.output_transform = output_transform
        #self.output_unit = get_torch_activation(output_unit) if output_unit else None 
        #self.output_layer = nn.Sequential(
        #    nn.Conv1d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=1),
        #    nn.Linear(in_features=self.timesamples, out_features=self.output_width)
        #)

    def forward(self, x):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified. 
        """
        x = x.permute(2,0,1)
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.permute(1,2,0)
        #_,transformer_reduc = self.reduction_head(transformer_out)
        # shape (bs, input_channels, timesamples)

        output = torch.flatten(transformer_out, start_dim=1)
        output = self.output_layer(output)

        #if self.output_width == 1: # squeeze (bs, 1, 1) to (bs, 1)
        #    output = output.squeeze(dim=2)
        #elif self.output_channels == 1: # squeee (bs, 1, 2) to (bs, 2)
        #    output = output.squeeze(dim=1)

        return output

    
    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network 
        nb.features has to be defined in a model implementing ConvNet
        """
        return self.input_channels * self.timesamples 



class ReductionHead(nn.Sequential):
    def __init__(self, nb_channels, emb_size):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(nb_channels),
            nn.Linear(nb_channels, emb_size)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out


if __name__ == "__main__":
    batch, chan, time = 16, 129, 500
    out_chan, out_width = 500, 3
    model = TransformerSimple(input_shape=(time, chan), output_shape=(out_chan, out_width))
    tensor = torch.randn(batch, chan, time)
    out = model(tensor)
    print(f"output shape: {out.shape}")