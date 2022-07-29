import tensorflow as tf
from config import config
from utils.utils import *
import logging
from DL_Models.tf_models.ConvNet import ConvNet
from tensorflow.keras.constraints import max_norm


class PyramidalCNN(ConvNet):
    """
    The Classifier_PyramidalCNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a
    specific depth, where for each depth the number of filters is increased.
    """
    def __init__(self, batch_size, input_shape, loss='mse', model_number=1,  output_shape=1, kernel_size=16, epochs = 50, nb_filters=16, verbose=True,
                    use_residual=False, depth=6):

        super(PyramidalCNN, self).__init__(input_shape=input_shape, output_shape=output_shape, loss=loss, kernel_size=kernel_size, epochs=epochs, nb_filters=nb_filters,
                    verbose=verbose, batch_size=batch_size, use_residual=use_residual, depth=depth, model_number=model_number)

    def __str__(self):
        return self.__class__.__name__
        
    def _module(self, input_tensor, current_depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is also used.
        """
        x = tf.keras.layers.Conv1D(filters=self.nb_filters*(current_depth + 1), kernel_size=self.kernel_size, padding='same', use_bias=False)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        x = tf.keras.layers.Add()([shortcut_y, out_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def buildModel(self,inputShape=None):
        input_layer = tf.keras.layers.Input(self.input_shape)

        if self.preprocessing:
            preprocessed = self._preprocessing(input_layer)
            x = preprocessed
            input_res = preprocessed
        else:
            x = input_layer
            input_res = input_layer

        for d in range(self.depth):
            x = self._module(x, d)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        if self.loss == 'bce':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='sigmoid')(gap_layer)
        elif self.loss == 'mse':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='linear')(gap_layer)
        elif self.loss == 'angle-loss':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='linear')(gap_layer)
        else:
            raise ValueError("Choose valid loss function")

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        return model
