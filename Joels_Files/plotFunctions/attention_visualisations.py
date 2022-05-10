import numpy as np
from config import config
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def saliencyMap(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, layer: int = -1, normalizeBool: bool = True) -> np.ndarray:
    if config['framework'] == 'tensorflow':
        return saliencyMapTensorflow(model, inputSignals, groundTruth, loss, layer)



def saliencyMapTensorflow(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str ,layer: int = -1, normalizeBool: bool = True) -> np.ndarray:
    import tensorflow.keras as keras
    import tensorflow as tf

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
        with tf.GradientTape() as g:
            g.watch(inputSignals)
            outputs = model(inputSignals,training=False)
            lossValues = loss(groundTruth,outputs)

            grads = keras.backend.abs(g.gradient(lossValues,inputSignals))
    else:
        auxModel = keras.Model(inputs=model.inputs, outputs=[model.layers[layer].output])
        with tf.GradientTape() as g:
            g.watch(inputSignals)
            outputs = auxModel(inputSignals,training=False)

            grads = keras.backend.abs(g.gradient(outputs,inputSignals))
    grads = grads.numpy()
    if normalizeBool:
        for i in range(grads.shape[0]):
            grads[i] = grads[i] / np.max(grads[i])


    model.layers[layer].activation = oldActivationType

    return grads


def plotSaliencyMap(inputSignals: np.ndarray, groundTruth: np.ndarray, gradients: np.ndarray, directory: str,
                  electrodesUsedForTraining: np.ndarray = np.arange(1, 130),electrodesToPlot: np.ndarray = np.arange(1, 130),
                  filename: str = 'AttentionVisualisation', format: str = 'pdf',maxValue: float = 100, saveBool: bool = True):

    indicesToPlot = findElectrodeIndices(electrodesUsedForTraining,electrodesToPlot)

    linSpace = np.arange(1, inputSignals.shape[1]*2+1, 2)
    for i in range(inputSignals.shape[0]):
        for j in indicesToPlot:
            f, ax = plt.subplots()
            ax.title.set_text(
                "Electrode {} of Sample {}".format(electrodesUsedForTraining[j], str(i)) + ", GroundTruth: " + np.array2string(
                    groundTruth[i]))
            ax.set_ylim([-maxValue, maxValue])
            ax.get_xaxis().set_major_formatter(FormatStrFormatter('%d ms'))
            ax.get_yaxis().set_major_formatter(FormatStrFormatter('%d mv'))
            ax.plot(linSpace, inputSignals[i, :, j], c='black')
            ax.imshow(np.repeat(np.repeat(np.expand_dims(gradients[i, :, j], axis=0), 2, axis=1), int(maxValue * 2), axis=0),
                      extent=[0, 1000, int(maxValue), -int(maxValue)], cmap='jet', alpha=0.5, origin='lower')
            if saveBool:
                plt.savefig(os.path.join(directory,
                                         filename + '_Sample{}_El{}.'.format(str(i), electrodesUsedForTraining[j]) + format))
            else:
                return plt
            plt.close()



###############Helper Functions#####################

def findElectrodeIndices(electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                         electrodesToPlot: np.ndarray = np.arange(1, 130)) -> np.ndarray:
    intersect, ind_a, electrodes = np.intersect1d(electrodesToPlot, electrodesUsedForTraining, return_indices=True)
    del intersect, ind_a

    return np.atleast_1d(electrodes)