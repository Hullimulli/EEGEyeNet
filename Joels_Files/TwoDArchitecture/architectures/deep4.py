from tensorflow import keras
import tensorflow as tf


def convPoolBlockOneShallow(previousLayer, nrOfFiltersTemporal: int = 40, nrOfFiltersSpatial: int = 40,
                            kernelSizeTemporal: int = 26,
                            kernelSizeSpatial: int = 129, stridePool: int = 15, meanPoolSize: int = 75, initializer = 'he_uniform'):
    x = keras.layers.Conv2D(filters=nrOfFiltersTemporal, kernel_size=(kernelSizeTemporal, 1), padding="valid",
                            use_bias=False,
                            kernel_initializer=initializer)(previousLayer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(filters=nrOfFiltersSpatial, kernel_size=(1, kernelSizeSpatial), padding="valid",
                            use_bias=False,
                            kernel_initializer=initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.backend.square(x)
    x = keras.layers.AveragePooling2D(pool_size=(meanPoolSize, 1), strides=(stridePool, 1), padding="valid")(x)
    x = keras.backend.log(x)
    return x

def convPoolBlockOne(previousLayer, nrOfFiltersTemporal: int = 25, nrOfFiltersSpatial: int = 25, kernelSizeTemporal: int = 10,
                      kernelSizeSpatial: int = 129, stridePool: int = 1, maxPoolSize: int = 3, initializer = 'he_uniform'):

    x = keras.layers.Conv2D(filters=nrOfFiltersTemporal, kernel_size=(kernelSizeTemporal,1),padding="valid", use_bias=False,
                            kernel_initializer=initializer)(previousLayer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(filters=nrOfFiltersSpatial, kernel_size=(1,kernelSizeSpatial), strides=(stridePool,1), padding="valid", use_bias=False,
                            kernel_initializer=initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("elu")(x)
    x = keras.layers.MaxPool2D(pool_size=(maxPoolSize,1), padding="valid")(x)

    return x

def convPoolBlockTwo(previousLayer, nrOfFilters: int = 50, kernelSize: int = 10, stridePool: int = 1, maxPoolSize: int = 3, initializer = 'he_uniform'):
    x = keras.layers.Dropout(0.5)(previousLayer)
    x = keras.layers.Conv2D(filters=nrOfFilters, kernel_size=(kernelSize,1), strides=(stridePool,1), padding="valid", use_bias=False,
                            kernel_initializer=initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("elu")(x)
    x = keras.layers.MaxPool2D(pool_size=(maxPoolSize,1), padding="valid")(x)

    return x


class Shallow4Net:
    """
    Shallow ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
    ):
        self.initializer = 'he_uniform'

    def buildModel(self, inputShape: (int ,int),loss,outputActivation,outputShape):
        inputs = keras.Input(shape=(inputShape[0], inputShape[1], 1))
        x = inputs
        x = convPoolBlockOneShallow(x,kernelSizeSpatial=inputShape[1])
        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(outputShape, activation=outputActivation,
                               kernel_initializer=self.initializer)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=loss)

        return model

class Deep4Net:
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
    ):
        self.initializer = 'he_uniform'

    def buildModel(self, inputShape: (int ,int ,int),loss,outputActivation,outputShape):
        inputs = keras.Input(shape=(inputShape[0], inputShape[1], 1))
        x = inputs
        x = convPoolBlockOne(x,kernelSizeSpatial=inputShape[1])
        x = convPoolBlockTwo(x,nrOfFilters=50)
        x = convPoolBlockTwo(x, nrOfFilters=100)
        x = convPoolBlockTwo(x, nrOfFilters=200)
        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(outputShape, activation=outputActivation,
                               kernel_initializer=self.initializer)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=loss)

        return model


class Hybrid4Net:
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self, learningRate = 1e-4
    ):
        self.initializer = 'he_uniform'
        self.learningRate = learningRate

    def buildModel(self, inputShape: (int ,int ,int),loss,outputActivation,outputShape):
        inputs = keras.Input(shape=(inputShape[0], inputShape[1], 1))
        x = inputs

        xDeep = convPoolBlockOne(x,kernelSizeSpatial=inputShape[1])
        xDeep = convPoolBlockTwo(xDeep,nrOfFilters=50)
        xDeep = convPoolBlockTwo(xDeep, nrOfFilters=100)
        xDeep = convPoolBlockTwo(xDeep, nrOfFilters=200,maxPoolSize=2)
        xDeep = keras.layers.Dropout(0.5)(xDeep)
        xDeep = keras.layers.Conv2D(filters=60, kernel_size=(2,1), padding="valid",
                                use_bias=True,
                                kernel_initializer=self.initializer)(xDeep)

        xShallow = convPoolBlockOneShallow(x,kernelSizeSpatial=inputShape[1])
        xShallow = keras.layers.Dropout(0.5)(xShallow)
        xShallow = keras.layers.Conv2D(filters=40, kernel_size=(27,1), padding="valid",
                                    use_bias=True,
                                    kernel_initializer=self.initializer)(xShallow)
        x = keras.layers.concatenate([xDeep,xShallow])
        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(outputShape, activation=outputActivation,
                               kernel_initializer=self.initializer)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningRate),loss=loss)

        return model
