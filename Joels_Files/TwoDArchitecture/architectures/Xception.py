import tensorflow as tf
from config import config
from utils.utils import *
import logging
import tensorflow.keras as keras



class XCEPTION():
    """
    The Xception architecture. This is inspired by Xception paper, which describes how 'extreme' convolutions can be represented
    as separable convolutions and can achieve better accuracy then the Inception architecture. It is made of modules in a specific depth.
    Each module, in our implementation, consists of a separable convolution followed by batch normalization and a ReLu activation layer.
    """
    def __init__(self, kernelSize=40, convFilters=128, residualBool=True, depth=6):
        self.convFilters = convFilters
        self.initializer = 'he_uniform'
        self.residualBool = residualBool
        self.kernelSize = kernelSize
        self.depth = depth

    def _module(self, input_tensor, current_depth):
        """
        The module of Xception. Consists of a separable convolution followed by batch normalization and a ReLu activation function.
        """
        x = tf.keras.layers.SeparableConv1D(filters=self.convFilters, kernel_size=self.kernelSize, padding='same', use_bias=False, depth_multiplier=1)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x


    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def buildModel(self, inputShape: (int, int), loss, outputActivation, outputShape):
        inputLayer = tf.keras.layers.Input(inputShape)
        x = inputLayer
        input_res = inputLayer
        for d in range(self.depth):
            x = self._module(x, d)
            if self.residualBool and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(outputShape, activation=outputActivation)(gap_layer)

        model = tf.keras.models.Model(inputs=inputLayer, outputs=output_layer)

        model.compile(optimizer="adam", loss=loss)
        return model