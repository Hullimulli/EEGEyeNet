import tensorflow as tf
from config import config
from utils.utils import *
import logging
import tensorflow.keras as keras


class INCEPTION():
    """
    The InceptionTime architecture used as baseline. This is the architecture explained in the paper

    'InceptionTime: Finding AlexNet for Time Series Classification' with authors
    Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier,
    Daniel F. Schmidt, Jonathan Weber, Geoffrey I. Webb, Lhassane Idoumghar, Pierre-Alain Muller, François Petitjean
    """

    def __init__(self, kernelSize=64, convFilters=16, residualBool=True, depth=12, bottleneckSize=16):
        self.bottleneckSize = bottleneckSize
        self.convFilters = convFilters
        self.initializer = 'he_uniform'
        self.residualBool = residualBool
        self.kernelSize = kernelSize
        self.depth = depth
        
    def _module(self, input_tensor, current_depth):
        """
        The module of InceptionTime (Taken from the implementation of InceptionTime paper).
        It is made of a bottleneck convolution that reduces the number of channels from 128 -> 32.
        Then it uses 3 filters with different kernel sizes (Default values are 40, 20, and 10)
        In parallel it uses a simple convolution with kernel size 1 with max pooling for stability during training.
        The outputs of each convolution are concatenated, followed by batch normalization and a ReLu activation.
        """
        if int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneckSize, kernel_size=1, padding='same', use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernelSize // (2 ** i) for i in range(3)]
        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(filters=self.convFilters, kernel_size=kernel_size_s[i], padding='same', use_bias=False)(input_inception))

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(input_tensor)
        conv_6 = tf.keras.layers.Conv1D(filters=self.convFilters, kernel_size=1, padding='same', use_bias=False)(max_pool_1)

        conv_list.append(conv_6)
        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
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
