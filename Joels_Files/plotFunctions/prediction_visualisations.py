import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import matplotlib.cm as cm

def visualizePredictionLR(groundTruth: np.ndarray, directory: str, prediction: np.ndarray,
               modelNames: list, filename: str = 'predictionVisualisation',
               format: str = 'pdf', saveBool: bool = True, colourMap: str = "nipy_spectral"):

    #Checks
    groundTruth = groundTruth.ravel()

    if prediction.ndim != 4:
        print("Need a 4 dimensional array as prediction.")
        return
    if groundTruth.shape[0] != prediction.shape[2]:
        print("Shapes of predictions and ground truths do not coincide.")
        return
    if len(modelNames) != prediction.shape[0]:
        print("Shapes of predictions and modelNames do not coincide.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "nipy_spectral"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    pass
    #TODO

def visualizePredictionPosition(groundTruth: np.ndarray, directory: str, prediction: np.ndarray,
                modelNames: list, filename: str = 'predictionVisualisation',
                format: str = 'pdf', saveBool: bool = True, colourMap: str = "nipy_spectral",
                BoundariesX: (int,int) = (0,800), BoundariesY: (int,int) = (0,600)):
    """
    Maps the ground truth to the euclidean space and plots the average prediction of all given architecture types.
    The variance is visualised with a circle.

    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth. Shape is
    [#architectures,#models,#predictions,2]
    @type prediction: Numpy Array
    @param modelNames: List of the names of all models.
    @type modelNames: List of Strings
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param colourMap: Matplotlib colour map for the plot.
    @type colourMap: String
    @param BoundariesX: Tuple of Integers, which determines the lower and upper bound of the area of the x-axis of the plot.
    @type BoundariesX: (Integer,Integer)
    @param BoundariesY: Tuple of Integers, which determines the lower and upper bound of the area of the y-axis of the plot.
    @type BoundariesY: (Integer,Integer)
    """


    #Checks
    if prediction.ndim != 4:
        print("Need a 4 dimensional array as prediction.")
        return
    if groundTruth.shape[0] != prediction.shape[2] or groundTruth.shape[1] != prediction.shape[3]:
        print("Shapes of predictions and ground truths do not coincide.")
        return
    if len(modelNames) != prediction.shape[0]:
        print("Shapes of predictions and modelNames do not coincide.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "nipy_spectral"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    cmap = cm.get_cmap(colourMap)
    colour = cmap((1 + np.arange(len(modelNames))) / len(modelNames))

    fig = plt.figure()
    plt.scatter(groundTruth[:, 0], groundTruth[:, 1], c='black', marker='x', label="Ground Truth")
    plt.axis('equal')
    plt.xlim([BoundariesX[0], BoundariesX[1]])
    plt.ylim([BoundariesY[0], BoundariesY[1]])
    plt.gca().invert_yaxis()
    for i, modelName in enumerate(modelNames):
        x = np.mean(prediction[i,:, :, 0], axis=0)
        y = np.mean(prediction[i,:, :, 1], axis=0)
        colours = np.zeros([x.shape[0], 4]) + colour[i]
        plt.scatter(x, y, s=5, c=colours, marker='o', label=modelName)
        radi = np.sqrt(np.square(np.std(prediction[i,:, :, 0], axis=0)) +
                       np.square(np.std(prediction[i,:, :, 1], axis=0)))
        ax = fig.gca()

        for j in range(x.shape[0]):
            plt.plot(np.array([x[j], groundTruth[j, 0]]), np.array([y[j], groundTruth[j, 1]]), c=colour[i])
        colour[i, 3] = 0.35
        for j in range(x.shape[0]):
            ax.add_patch(plt.Circle((x[j], y[j]), radi[j], color=colour[i]))

    plt.axhline(0, color='black', linewidth=0.1)
    plt.axvline(0, color='black', linewidth=0.1)

    plt.legend()
    if saveBool:
        fig.savefig(os.path.join(directory,filename) + ".{}".format(format), format=format, transparent=True)
    else:
        plt.show()
    plt.close()

def visualizePredictionAngle(groundTruth: np.ndarray, directory: str, prediction: np.ndarray,
               modelNames: list, filename: str = 'predictionVisualisation',
               format: str = 'pdf', saveBool: bool = True, colourMap: str = "nipy_spectral"):

    #Checks
    groundTruth = groundTruth.ravel()
    if prediction.ndim != 4:
        print("Need a 4 dimensional array as prediction.")
        return
    if groundTruth.shape[0] != prediction.shape[2]:
        print("Shapes of predictions and ground truths do not coincide.")
        return
    if len(modelNames) != prediction.shape[0]:
        print("Shapes of predictions and modelNames do not coincide.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "nipy_spectral"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    pass
    # TODO

def visualizePredictionAmplitude(groundTruth: np.ndarray, directory: str, prediction: np.ndarray,
               modelNames: list, filename: str = 'predictionVisualisation',
               format: str = 'pdf', saveBool: bool = True, colourMap: str = "nipy_spectral"):

    #Checks
    groundTruth = groundTruth.ravel()
    if prediction.ndim != 4:
        print("Need a 4 dimensional array as prediction.")
        return
    if groundTruth.shape[0] != prediction.shape[2]:
        print("Shapes of predictions and ground truths do not coincide.")
        return
    if len(modelNames) != prediction.shape[0]:
        print("Shapes of predictions and modelNames do not coincide.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "nipy_spectral"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    pass
    # TODO