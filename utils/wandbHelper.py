import numpy as np
import wandb
from Joels_Files.plotFunctions.prediction_visualisations import visualizePredictionLR,visualizePredictionAngle,visualizePredictionAmplitude,visualizePredictionPosition
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
        logs = {"Prediction Visualisation": wandb.Image(visualizePredictionAngle(groundTruth=groundTruth[:10],
                                                                 prediction=np.expand_dims(prediction[:10], axis=(0, 1)),
                                                                 modelNames=[modelName],directory="./",saveBool=False,
                                                                 colourMap="cool"))}
    elif loss == 'bce':
        logs = {"Prediction Visualisation": wandb.Image(visualizePredictionLR(groundTruth=groundTruth[:30],
                                                                 prediction=np.expand_dims(prediction[:30], axis=(0, 1)),
                                                                 modelNames=[modelName],directory="./",saveBool=False,
                                                                 colourMap="cool"))}
    elif groundTruth.shape[1] == 1:
        logs = {"Prediction Visualisation": wandb.Image(visualizePredictionAmplitude(groundTruth=groundTruth[:10],
                                                                 prediction=np.expand_dims(prediction[:10], axis=(0, 1)),
                                                                 modelNames=[modelName],directory="./",saveBool=False,
                                                                 colourMap="cool"))}
    else:
        logs = {"Prediction Visualisation": wandb.Image(visualizePredictionPosition(groundTruth=groundTruth[:10],
                                                                 prediction=np.expand_dims(prediction[:10], axis=(0, 1)),
                                                                 modelNames=[modelName],directory="./",saveBool=False,
                                                                 colourMap="cool"))}
    plt.close('all')

    return {**logs, **saliencyMaps}