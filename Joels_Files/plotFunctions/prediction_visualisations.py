import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
from config import config

def getVisualisation(groundTruth: np.ndarray, prediction: np.ndarray, modelName: str,anglePartBool: bool = False):
    if config['task'] == "LR_task":
        return visualizePredictionLR(groundTruth=groundTruth[:30],prediction=prediction[:,:,:30],modelNames=[modelName],
                                     directory="./",saveBool=False, colourMap="cool")
    elif config['task'] == "Direction_task":
        if not anglePartBool:
            return visualizePredictionAmplitude(groundTruth=groundTruth[:10],prediction=prediction[:,:,:10],
                                                modelNames=[modelName],directory="./",saveBool=False, colourMap="cool")
        return visualizePredictionAngle(groundTruth=groundTruth[:10],prediction=prediction[:,:,:10],
                                        modelNames=[modelName],directory="./",saveBool=False, colourMap="cool")
    elif config['task'] == "Position_task":
        return visualizePredictionPosition(groundTruth=groundTruth[:10],prediction=prediction[:,:,:10],
                                           modelNames=[modelName],directory="./",saveBool=False, colourMap="cool")
    else:
        print("Task visualisation not yet implemented.")
        return

def visualizePredictionLR(groundTruth: np.ndarray, prediction: np.ndarray, modelNames: list,
            directory: str, filename: str = 'predictionVisualisation',
            format: str = 'pdf', saveBool: bool = True, colourMap: str = "nipy_spectral"):

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
    if not np.all(np.logical_or(groundTruth == 0, groundTruth == 1)):
        print("Ground truth contains unforseen labels.")

    lefts = np.atleast_1d(np.squeeze(np.argwhere(groundTruth == 0)))
    rights = np.atleast_1d(np.squeeze(np.argwhere(groundTruth == 1)))
    cmap = cm.get_cmap(colourMap)
    colour = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    yAxis = np.arange(groundTruth.shape[0]) / 2

    fig = plt.figure()
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off
    x_patch = mlines.Line2D([], [],color='black', label='GT Right', marker='x',linestyle='None')
    o_patch = mlines.Line2D([], [],color='black', label='GT Left', marker='o',linestyle='None')
    plt.ylim(-0.1,groundTruth.shape[0]/2-0.45)
    plt.xlim(-0.1,1.1)
    unitType = np.int
    if len(modelNames) == 1:
        unitType = np.float
    for i, modelName in enumerate(modelNames):
        plt.scatter((i + 1)*(np.mean(prediction[i, :, lefts]+0.5,axis=1).astype(unitType)-np.array(0.5).astype(unitType))/len(modelNames),
                    yAxis[lefts],  color=colour[i], marker='o',label=modelName)
        plt.scatter((i + 1)*(np.mean(prediction[i, :, rights]+0.5,axis=1).astype(unitType)-np.array(0.5).astype(unitType))/len(modelNames),
                    yAxis[rights],  marker='x', color=colour[i])
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([x_patch, o_patch])
    plt.legend(handles=handles)
    if saveBool:
        fig.savefig(os.path.join(directory,filename) + ".{}".format(format), format=format, transparent=True)
    else:
        return plt
    plt.close()
    del fig

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
    @param saveBool: If True, the plot will be saved. Else it will be returned.
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
        return plt
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
    @param saveBool: If True, the plot will be saved. Else it will be returned.
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
        return plt
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
    @param saveBool: If True, the plot will be saved. Else it will be returned.
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

    cmap = cm.get_cmap(colourMap)
    colour = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    colourLight = cmap((1 + np.arange(len(modelNames))) / len(modelNames))
    colourLight[:, 3] = 0.35

    fig = plt.figure()
    plt.scatter((np.arange(groundTruth.shape[0])) / groundTruth.shape[0]  - 0.5, groundTruth,
                c='black', marker='x', label="Ground Truth")
    plt.axis('auto')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d px'))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.xlim((-int(len(modelNames)/2) - 0.6, int((len(modelNames)+1)/2) + 0.6))
    plt.ylim(Boundaries)
    for i, modelName in enumerate(modelNames):
        # We want the ground truth to be in the middle.
        if i >= int(len(modelNames) / 2 + 1) - 1:
            pos = i + 0.5 - int(len(modelNames)/2)
        else:
            pos = i - 0.5 - int(len(modelNames)/2)
        y = np.mean(prediction[i, :, :], axis=0)
        x = np.arange(y.shape[0]) / groundTruth.shape[0] + pos
        sigmaAmp = np.std(prediction[i, :, :], axis=0)
        plt.errorbar(x, y, yerr=sigmaAmp, color=colour[i], label=modelName, fmt='o', linestyle="None")
        for j in range(x.shape[0]):
            plt.plot(np.array([x[j], (np.arange(groundTruth.shape[0])[j]) / groundTruth.shape[0] - 0.5]),
                     np.array([y[j], groundTruth[j]]), c=colourLight[i])

    plt.legend()
    if saveBool:
        fig.savefig(os.path.join(directory,filename) + ".{}".format(format), format=format, transparent=True)
    else:
        return plt
    plt.close()
    del fig