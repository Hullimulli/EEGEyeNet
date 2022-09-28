import tensorflow as tf
from config import config
from utils.utils import *
import logging
from DL_Models.tf_models.ConvNet import ConvNet
import tensorflow.keras as keras


class PyramidalCNN():
    """
    The Classifier_PyramidalCNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a
    specific depth, where for each depth the number of filters is increased.
    """
    def __init__(self, kernelSize=16, convFilters=16,
                    residualBool=False, depth=12):

        self.convFilters = convFilters
        self.initializer = 'he_uniform'
        self.residualBool = residualBool
        self.kernelSize = kernelSize
        self.depth = depth

    def __str__(self):
        return self.__class__.__name__
        
    def _module(self, input_tensor, current_depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is also used.
        """
        x = tf.keras.layers.Conv1D(filters=self.convFilters*(current_depth + 1), kernel_size=self.kernelSize, padding='same', use_bias=False)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
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