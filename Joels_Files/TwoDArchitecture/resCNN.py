import tensorflow.keras as keras
import tensorflow as tf

class resCNN:

    def __init__(self,filters: list = [128,256,512,1024]):
        self.filters = filters
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
        x = x = keras.layers.BatchNormalization()(x)
        x = keras.layers.add([residual,x])

        x = keras.layers.Activation("relu")(x)

        if reduceBool:
            x = keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(x)

        return x

    def buildModel(self, inputShape: (int ,int ,int), loss="mse"):
        inputs = keras.Input(shape=inputShape)
        x = inputs
        for filters in self.filters:
            x = self.downSamplingBlock(previousLayer=x ,nrOfFilters=filters)

        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1, activation="relu",
                                     kernel_initializer=self.initializer, use_bias=False)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam" ,loss=loss)

        return model