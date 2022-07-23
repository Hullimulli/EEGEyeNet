import tensorflow.keras as keras
import tensorflow as tf

class resCNN:

    def __init__(self,convFilters: list = [128,256,512], denseFilters: list = [1024,512,256,128]):
        self.convFilters = convFilters
        self.denseFilters = denseFilters
        self.initializer = 'he_uniform'

    def downSamplingBlock(self ,previousLayer, nrOfFilters: int, reduceBool: bool = True):

        residual = keras.layers.Conv2D(filters=nrOfFilters, kernel_size=1, padding="same", use_bias=False,
                                       kernel_initializer=self.initializer)(
            previousLayer)

        x = keras.layers.Conv2D(filters=nrOfFilters, kernel_size=3, padding="same", use_bias=False,
                                kernel_initializer=self.initializer)(previousLayer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Conv2D(filters=nrOfFilters, kernel_size=3, padding="same", use_bias=False,
                                kernel_initializer=self.initializer)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.add([residual,x])

        x = keras.layers.Activation("relu")(x)

        if reduceBool:
            x = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(x)

        return x

    def buildModel(self, inputShape: (int ,int ,int), loss="mse"):
        inputs = keras.Input(shape=inputShape)
        x = inputs
        for filters in self.convFilters:
            x = self.downSamplingBlock(previousLayer=x ,nrOfFilters=filters)

        x = keras.layers.GlobalAveragePooling2D()(x)
        for filters in self.denseFilters:
            x = keras.layers.Dense(filters, activation="relu",
                               kernel_initializer=self.initializer, use_bias=False)(x)
        outputs = keras.layers.Dense(1, activation="relu",
                               kernel_initializer=self.initializer, use_bias=False)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam" ,loss=loss)

        return model



class resCNN1D:

    def __init__(self,convFilters: list = [128,256,512], denseFilters: list = [1024,512,256,128]):
        self.convFilters = convFilters
        self.denseFilters = denseFilters
        self.initializer = 'he_uniform'

    def downSamplingBlock(self ,previousLayer, nrOfFilters: int, reduceBool: bool = True):

        residual = keras.layers.Conv1D(filters=nrOfFilters, kernel_size=1, padding="same", use_bias=False,
                                       kernel_initializer=self.initializer)(
            previousLayer)

        x = keras.layers.Conv1D(filters=nrOfFilters, kernel_size=64, padding="same", use_bias=False,
                                kernel_initializer=self.initializer)(previousLayer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Conv1D(filters=nrOfFilters, kernel_size=64, padding="same", use_bias=False,
                                kernel_initializer=self.initializer)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.add([residual,x])

        x = keras.layers.Activation("relu")(x)

        if reduceBool:
            x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)

        return x

    def buildModel(self, inputShape: (int ,int ,int), loss="mse"):
        inputs = keras.Input(shape=inputShape)
        x = inputs
        for filters in self.convFilters:
            x = self.downSamplingBlock(previousLayer=x ,nrOfFilters=filters)

        x = keras.layers.GlobalAveragePooling1D()(x)
        for filters in self.denseFilters:
            x = keras.layers.Dense(filters, activation="relu",
                               kernel_initializer=self.initializer, use_bias=False)(x)
        outputs = keras.layers.Dense(1, activation="relu",
                               kernel_initializer=self.initializer, use_bias=False)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam" ,loss=loss)

        return model