from torch import nn
import torch
from abc import ABC
import pytorch_lightning as pl 
from DL_Models.torch_models.BaseNetTorch import BaseNet


class biLSTM(ABC, BaseNet):

    def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
                use_residual=True, depth=12, hidden_size=129, dropout=0.5):
        """
        We define the layers of the network in the __init__ function
        """
        self.hidden_size = hidden_size
        self.timesamples = input_shape[0]
        self.input_channels = input_shape[1]
        super().__init__(model_name=model_name, path=path, model_number=model_number, loss=loss, input_shape=input_shape, output_shape=output_shape, 
                        epochs=epochs, verbose=verbose)
        
        #self.output_channels = output_shape[0]
        #self.output_width = output_shape[1]
        # Define the modules
        self.lstm = nn.LSTM(input_size=self.input_channels, hidden_size=self.hidden_size, dropout=dropout, 
                            num_layers=3, bidirectional=True)
        #self.output_conv = nn.Conv1d(in_channels=2*self.input_channels, out_channels=self.output_channels, kernel_size=1)
        #self.output_linear = nn.Linear(self.timesamples, self.output_width)

    def forward(self, x, hidden=None):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified.
        """

        x = x.permute(2, 0, 1)
        output, (hn, cn) = self.lstm(x, hidden)
        output = output.permute(1,2,0) 
        # shape (bs, 2*input_channels, timesamples)
        #output = self.output_conv(output) 
        # shape (bs, output_channels, timesamples)
        #output = self.output_linear(output) 
        # shape (bs, output_channels, output_width)
       # if self.output_width == 1: # squeeze (bs, 1, 1) to (bs, 1)
        #    output = output.squeeze(dim=2)
        #elif self.output_channels == 1: # squeee (bs, 1, 2) to (bs, 2)
         #   output = output.squeeze(dim=1)
        #return output

        output = torch.flatten(output, start_dim=1)
        output = self.output_layer(output)
        return output

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network
        nb.features has to be defined in a model implementing ConvNet
        """
        return 2 * self.hidden_size * self.timesamples 


if __name__ == "__main__":
    batch, chan, time = 16, 129, 500
    out_chan, out_width = 1,1
    model = biLSTM(input_shape=(time, chan), output_shape=(out_chan, out_width))
    tensor = torch.randn(batch, chan, time)
    out = model(tensor)
    print(f"output shape: {out.shape}")
