import tensorflow.keras as keras
import tensorflow as tf

class resCNN2D:

    def __init__(self,convFilters: list = [128,256,512], denseFilters: list = [], residualBool: bool = True):
        self.convFilters = convFilters
        self.denseFilters = denseFilters
        self.initializer = 'he_uniform'
        self.residualBool = residualBool

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

        if self.residualBool:
            x = keras.layers.add([residual,x])

        x = keras.layers.Activation("relu")(x)

        if reduceBool:
            x = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(x)

        return x

    def buildModel(self, inputShape: (int ,int ,int)):
        inputs = keras.Input(shape=inputShape)
        x = inputs
        for filters in self.convFilters:
            x = self.downSamplingBlock(previousLayer=x ,nrOfFilters=filters)

        x = keras.layers.GlobalAveragePooling2D()(x)
        for filters in self.denseFilters:
            x = keras.layers.Dense(filters, activation="relu",
                               kernel_initializer=self.initializer, use_bias=False)(x)
        outputs = keras.layers.Dense(1, activation="linear",
                               kernel_initializer=self.initializer)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam")

        return model



class CNN1D:

    def __init__(self,convFilters: list = [16,32,48,64,80,96], denseFilters: list = []):
        self.convFilters = convFilters
        self.denseFilters = denseFilters
        self.initializer = 'he_uniform'

    def downSamplingBlock(self ,previousLayer, nrOfFilters: int, reduceBool: bool = True):

        x = keras.layers.Conv1D(filters=nrOfFilters, kernel_size=16, padding="same", use_bias=False,
                                kernel_initializer=self.initializer)(previousLayer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        if reduceBool:
            x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)

        return x

    def buildModel(self, inputShape: (int ,int ,int)):
        inputs = keras.Input(shape=inputShape)
        x = inputs
        for filters in self.convFilters:
            x = self.downSamplingBlock(previousLayer=x ,nrOfFilters=filters)

        x = keras.layers.GlobalAveragePooling1D()(x)
        for filters in self.denseFilters:
            x = keras.layers.Dense(filters, activation="relu",
                               kernel_initializer=self.initializer, use_bias=False)(x)
        outputs = keras.layers.Dense(1, activation="linear",
                               kernel_initializer=self.initializer)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

        return model