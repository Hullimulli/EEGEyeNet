import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

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

    """
    Plots the average angle prediction of all given architectures. Radius is prediction independent.

    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth. Shape is
    [#architectures,#models,#predictions]
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
    """

    #Checks
    groundTruth = groundTruth.ravel()
    if prediction.ndim != 3:
        print("Need a 3 dimensional array as prediction.")
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

    #Plotting
    fig = plt.figure()
    plt.scatter(np.multiply(int(len(modelNames) / 2) + 1 + np.arange(groundTruth.shape[0]) / groundTruth.shape[0], np.cos(groundTruth)),
                np.multiply(int(len(modelNames) / 2) + 1 + np.arange(groundTruth.shape[0]) / groundTruth.shape[0], np.sin(groundTruth)),
                c='black', marker='x',
                label="Ground Truth")
    plt.axis('equal')
    plt.tick_params(
        axis='both',  # changes apply to both axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.yticks([])
    plt.gca().invert_yaxis()
    # Lines and labels for the angle
    centers = np.linspace(0, 2 * np.pi, 9)[:8]
    for j in range(8):
        plt.plot(np.array([0, (len(modelNames) + 2) * np.cos(centers[j])]),
                 np.array([0, (len(modelNames) + 2) * np.sin(centers[j])]), c='black', alpha=0.5)
        plt.text(len(modelNames) * np.cos(centers[j]), len(modelNames) * np.sin(centers[j]),
                 str(round((centers[j]) / np.pi * 180, 1)) + "°")

    cmap = cm.get_cmap(colourMap)
    colour = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    colourLight = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    colourLight[:, 3] = 0.35

    for i, modelName in enumerate(modelNames):
        rad = np.arange(groundTruth.shape[0]) / groundTruth.shape[0] + i + 1
        # The ground truth is at rad=int(len(ModelNames)/2)
        if i >= int(len(modelNames) / 2):
            rad += 1

        # To get only values between -pi & +pi. This step is unnecessary, but the code still is a bit clearer.
        predictions = np.arctan2(np.sin(prediction[i,:, :]), np.cos(prediction[i,:, :]))

        means = np.arctan2(np.mean(np.sin(predictions), axis=0), np.mean(np.cos(predictions), axis=0))
        y = rad * np.sin(means)
        x = rad * np.cos(means)
        diff = predictions - means
        sigmaAng = np.sqrt(np.mean(np.square(np.arctan2(np.sin(diff), np.cos(diff))), axis=0))
        plt.scatter(x, y, color=colour[i], marker='o', label=modelName)
        for j in range(x.shape[0]):
            theta = np.linspace(-sigmaAng[j] / 2, sigmaAng[j] / 2, 100) + means[j]
            plt.plot(rad[j] * np.cos(theta), rad[j] * np.sin(theta), c=colour[i])
        for j in range(x.shape[0]):
            plt.plot(np.array([x[j], np.multiply(int(len(modelNames) / 2) + 1 + j / groundTruth.shape[0], np.cos(groundTruth[j]))]),
                     np.array([y[j], np.multiply(int(len(modelNames) / 2) + 1 + j / groundTruth.shape[0], np.sin(groundTruth[j]))]),
                     c=colourLight[i])

    plt.legend()
    if saveBool:
        fig.savefig(os.path.join(directory,filename) + ".{}".format(format), format=format, transparent=True)
    else:
        plt.show()
    plt.close()

def visualizePredictionAmplitude(groundTruth: np.ndarray, directory: str, prediction: np.ndarray,modelNames: list,
                                 filename: str = 'predictionVisualisation',format: str = 'pdf', saveBool: bool = True,
                                 colourMap: str = "nipy_spectral", Boundaries: (int,int) = (0,1000)):
    """
    Plots the average amplitude prediction of all given architectures. X-axis is prediction independent.

    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth. Shape is
    [#architectures,#models,#predictions]
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
    @param Boundaries: Minimal and maximal amplitude of the plot.
    @type Boundaries: (Integer,Integer)
    """

    #Checks
    groundTruth = groundTruth.ravel()
    if prediction.ndim != 3:
        print("Need a 3 dimensional array as prediction.")
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

    cmap = cm.get_cmap('nipy_spectral')
    colour = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    colourLight = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    colourLight[:, 3] = 0.35

    fig = plt.figure()
    plt.scatter((np.arange(groundTruth.shape[0]) - len(modelNames) / 2) / groundTruth.shape[0], groundTruth,
                c='black', marker='x', label="Ground Truth")
    plt.axis('auto')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d px'))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.xlim((-len(modelNames) / 2 - 1.5, len(modelNames) / 2 + 1.5))
    plt.ylim(Boundaries)
    for i, modelName in enumerate(modelNames):
        # We want the ground truth to be in the middle.
        if i < len(modelNames) / 2:
            pos = i + 1
        else:
            pos = i - len(modelNames)
        y = np.mean(prediction[i, :, :], axis=0)
        x = (np.arange(y.shape[0]) - len(modelNames) / 2) / groundTruth.shape[0] + pos
        sigmaAmp = np.std(prediction[i, :, :], axis=0)
        plt.errorbar(x, y, yerr=sigmaAmp, color=colour[i], fmt='.k', label=modelName)
        for j in range(x.shape[0]):
            plt.plot(np.array([x[j], (np.arange(groundTruth.shape[0])[j] - len(modelNames) / 2) / groundTruth.shape[0]]),
                     np.array([y[j], groundTruth[j]]), c=colourLight[i])

    plt.legend()
    if saveBool:
        fig.savefig(os.path.join(directory,filename) + ".{}".format(format), format=format, transparent=True)
    else:
        plt.show()
    plt.close()
    del fig