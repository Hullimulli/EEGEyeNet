import numpy as np
import wandb
from Joels_Files.plotFunctions.prediction_visualisations import getVisualisation
from Joels_Files.plotFunctions.attention_visualisations import saliencyMap, plotSaliencyMap
import matplotlib.pyplot as plt


def getPredictionVisualisationsSimple(modelName: str, groundTruth: np.ndarray, prediction: np.ndarray, loss: str):

    groundTruth = np.atleast_1d(np.squeeze(groundTruth))
    prediction = np.atleast_1d(np.squeeze(prediction))

    if loss == "angle-loss":
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                         prediction=np.expand_dims(prediction,
                                                                                                   axis=(0, 1)),
                                                                         modelName=modelName, anglePartBool=True))}
    elif loss == 'bce':
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                         prediction=np.expand_dims(prediction,
                                                                                                   axis=(0, 1)),
                                                                         modelName=modelName, anglePartBool=False))}
    else:
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                         prediction=np.expand_dims(prediction,
                                                                                                   axis=(0, 1)),
                                                                         modelName=modelName, anglePartBool=False))}
    plt.close('all')

    return logs

def getPredictionVisualisations(model, modelName: str, inputSignals: np.ndarray, groundTruth: np.ndarray, prediction: np.ndarray,loss: str,
                                electrodesToPlot: np.ndarray = np.array([1,32,125,128]),
                                electrodesUsed: np.ndarray = np.arange(1, 130), preprocess = lambda x: x,
                                inversePreprocess = lambda x: x):
    groundTruth = np.atleast_1d(np.squeeze(groundTruth))
    prediction = np.atleast_1d(np.squeeze(prediction))

    grads = saliencyMap(model,preprocess(inputSignals[[0]]),groundTruth[[0]],loss,includeInputBool=True)
    grads = inversePreprocess(grads)
    saliencyMaps = dict()
    for e in electrodesToPlot:
        saliencyMaps['Attention Electrode '+str(e)] = wandb.Image(plotSaliencyMap(inputSignals[[0]],groundTruth[[0]],grads,
                                                                directory='',
                                                                electrodesUsedForTraining=electrodesUsed,
                                                                electrodesToPlot=np.atleast_1d(e),
                                                                saveBool=False))

    if loss == "angle-loss":
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                 prediction=np.expand_dims(prediction, axis=(0, 1)),
                                                                 modelName=modelName, anglePartBool=True))}
    elif loss == 'bce':
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                 prediction=np.expand_dims(prediction, axis=(0, 1)),
                                                                 modelName=modelName, anglePartBool=False))}
    else:
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                 prediction=np.expand_dims(prediction, axis=(0, 1)),
                                                                 modelName=modelName, anglePartBool=False))}
    plt.close('all')

    return {**logs, **saliencyMaps}