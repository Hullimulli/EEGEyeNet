import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def saliencyMap(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, layer: int = -1):
    if type(model) is None:
        saliencyMapTensorflow(model, inputSignals, groundTruth, loss, layer)



def saliencyMapTensorflow(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str ,layer: int = -1, nrOfActivations: int = 9):

    oldActivationType = model.layers[layer].activation
    model.layers[layer].activation = keras.activations.linear

    if loss == "bce":
        loss = lambda y,yPred: tf.multiply(-2*(y-0.5),yPred)
    elif loss == "mse":
        loss = lambda y,yPred: tf.norm(y - yPred, axis=1)
    elif loss == "angle-loss":
        loss = lambda y,yPred: tf.sqrt(tf.square(tf.math.atan2(tf.sin(yPred - y), tf.cos(yPred - y))))

    inputSignals = tf.convert_to_tensor(inputSignals)
    if layer == -1:
        nrOfActivations = 1
        with tf.GradientTape() as g:
            g.watch(inputSignals)
            outputs = model(inputSignals)
            lossValues = loss(groundTruth,outputs)

            grads = g.gradient(lossValues,inputSignals)
    else:
        auxModel = keras.Model(inputs=model.inputs, outputs=[model.layers[layer].output])
        with tf.GradientTape() as g:
            g.watch(inputSignals)
            outputs = auxModel(inputSignals)

            grads = g.gradient(outputs,inputSignals)

        #Sum
