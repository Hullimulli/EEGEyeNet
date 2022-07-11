from DL_Models.torch_models.ConvNetTorch import ConvNet
import torch.nn as nn
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool

class CNN(ConvNet):
    """
    The CNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a specific depth.
    """
    def __init__(self, model_name, path, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
                use_residual=True, depth=12,saveModel_suffix=''):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.saveModel_suffix = saveModel_suffix
        self.nb_features = nb_filters # For CNN simply the number of filters / channels 
        super().__init__(model_name=model_name, path=path, loss=loss, model_number=model_number, batch_size=batch_size, input_shape=input_shape, 
                            output_shape=output_shape, kernel_size=kernel_size, epochs=epochs, nb_filters=nb_filters, verbose=verbose,
                            use_residual=use_residual, depth=depth)

    def _module(self, depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is used.
        We use two custom padding modules such that keras-like padding='same' is achieved, i.e. tensor shape stays constant when passed through the module.
        """
        return nn.Sequential(
                Pad_Conv(kernel_size=self.kernel_size, value=0),
                nn.Conv1d(in_channels=self.nb_channels if depth==0 else self.nb_features, 
                            out_channels=self.nb_features, kernel_size=self.kernel_size, bias=False),
                nn.BatchNorm1d(num_features=self.nb_features),
                nn.ReLU(), # leakyrelu
                Pad_Pool(left=0, right=1, value=0),
                nn.MaxPool1d(kernel_size=2, stride=1)
                )


if __name__ == '__main__':
    model = CNN(model_name='CNN', path='', loss='mse', model_number=1, batch_size=32, input_shape=(500, 129), output_shape=2, kernel_size=64, epochs = 50, nb_filters=64, verbose=True,
                use_residual=True, depth=12,saveModel_suffix='')
    print(model)