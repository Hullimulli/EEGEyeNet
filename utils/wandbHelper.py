import numpy as np
import wandb
from Joels_Files.plotFunctions.prediction_visualisations import getVisualisation
import matplotlib.pyplot as plt




def getPredictionVisualisations(modelName: str, groundTruth: np.ndarray, prediction: np.ndarray,loss: str):
    groundTruth = np.squeeze(groundTruth)
    prediction = np.squeeze(prediction)

    if loss == "angle-loss":
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                 prediction=np.expand_dims(prediction, axis=(0, 1)),
                                                                 modelName=modelName, anglePartBool=True))}
    else:
        logs = {"Prediction Visualisation": wandb.Image(getVisualisation(groundTruth=groundTruth,
                                                                 prediction=np.expand_dims(prediction, axis=(0, 1)),
                                                                 modelName=modelName, anglePartBool=False))}
    plt.close('all')

    return logs