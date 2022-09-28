import tensorflow as tf
import tensorflow.keras as keras
from config import config
from utils.utils import *
from DL_Models.tf_models.BaseNet import BaseNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import numpy as np
import logging


class EEGNet():
    """
    The EEGNet architecture used as baseline. This is the architecture explained in the paper

    'EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces' with authors
    Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, Brent J. Lance
    """

    def __init__(self,
                F1 = 16, F2 = 256, D=4, kernel_size=250, dropout_rate = 0.5,
                norm_rate = 0.5, dropoutType = 'Dropout'):

        #self.nb_classes = nb_classes
        self.dropoutRate = dropout_rate
        self.kernLength = kernel_size
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.dropoutType = dropoutType


    def buildModel(self, inputShape: (int, int), loss, outputActivation, outputShape):
        """
        The model of EEGNet (Taken from the implementation of EEGNet paper).
        """
        if self.dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        
        input1 = Input(shape=(inputShape[0], inputShape[1], 1))

        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same',
                        input_shape=(inputShape[0], inputShape[1], 1),
                        use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((inputShape[0], 1), use_bias=False,
                                 depth_multiplier=self.D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 16))(block1)
        block1 = dropoutType(self.dropoutRate)(block1)

        block2 = SeparableConv2D(self.F2, (1, 64),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 6))(block2)
        block2 = dropoutType(self.dropoutRate)(block2)

        flatten = Flatten()(block2)

        # Create output layer depending on task
        output_layer = tf.keras.layers.Dense(outputShape, activation=outputActivation)(flatten)

        model = tf.keras.models.Model(inputs=input1, outputs=output_layer)

        model.compile(optimizer="adam", loss=loss)
        return model
