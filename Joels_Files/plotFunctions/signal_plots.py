import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


def plotSignal(inputSignals: np.ndarray, groundTruth: np.ndarray,directory: str, prediction: np.ndarray = None,
               electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
               electrodesToPlot: np.ndarray = np.arange(1, 130), filename: str = 'SignalVisualisation',
               format: str = 'pdf', saveBool: bool = True,maxValue: float = 100):
    """
    Visualises the signals.

    @param inputSignals: 3d Tensor of the signals which have to be plotted. Shape has to be [#samples,#timestamps,
    #electrodes]
    @type inputSignals: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param electrodesUsedForTraining: Electrodes on which the network has been trained. These are the electrodes which
    correspond to the indices in the last dimension.
    @type electrodesUsedForTraining: Numpy Array
    @param electrodesToPlot: Electrodes for which the plots are to be generated.
    @type electrodesToPlot: Numpy Array
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param maxValue: The maximum value in mV which is shown.
    @type maxValue: Float
    """
    plotSignalLR(inputSignals=inputSignals, groundTruth=groundTruth, directory=directory, prediction=prediction,
                 electrodesUsedForTraining=electrodesUsedForTraining, electrodesToPlot=electrodesToPlot,
                 filename=filename, format=format, saveBool=saveBool, maxValue=maxValue,
                 plotSignalsSeperatelyBool=True, meanBool=False)

def plotHorizontalEyeMovement(eyeMovement: np.ndarray, groundTruth: np.ndarray,directory: str,
                              prediction: np.ndarray = None, filename: str = 'SignalVisualisation',
                              format: str = 'pdf', saveBool: bool = True, maxValue: int = 800):
    """
    Visualises the horizontal eye movement.

    @param eyeMovement: 2d Tensor of the eye movements which have to be plotted. Shape has to be [#samples,#timestamps]
    @type eyeMovement: Numpy Array
    @param groundTruth: Ground truth label corresponding to the eye movement.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param maxValue: The maximum value in mV which is shown.
    @type maxValue: Integer
    """
    plotEyeLR(eyeMovement=eyeMovement, groundTruth=groundTruth, directory=directory, prediction=prediction,
                 filename=filename, format=format, saveBool=saveBool, maxValue=maxValue,
                 plotSignalsSeperatelyBool=True, meanBool=False)


def plotSignalLR(inputSignals: np.ndarray, groundTruth: np.ndarray, directory: str, prediction: np.ndarray = None,
                 electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                 electrodesToPlot: np.ndarray = np.arange(1, 130),
                 colourMap: str = 'gist_rainbow', misClasThresh: float = 0,
                 filename: str = 'SignalVisualisation', format: str = 'pdf', saveBool: bool = True,
                 plotSignalsSeperatelyBool: bool = False, maxValue: float = 100, meanBool: bool = True):

    """
    Visualises and colour codes the signals according to ground truth and prediction.

    @param inputSignals: 3d Tensor of the signals which have to be plotted. Shape has to be [#samples,#timestamps,
    #electrodes]
    @type inputSignals: Numpy Array
    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param electrodesUsedForTraining: Electrodes on which the network has been trained. These are the electrodes which
    correspond to the indices in the last dimension.
    @type electrodesUsedForTraining: Numpy Array
    @param electrodesToPlot: Electrodes for which the plots are to be generated.
    @type electrodesToPlot: Numpy Array
    @param colourMap: Matplotlib colour map for the plot.
    @type colourMap: String
    @param misClasThresh: Only plots signals which the difference between prediction and ground truth is larger
    than this.
    @type misClasThresh: Float
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param plotSignalsSeperatelyBool: If True, each signal is plotted in a new window.
    @type plotSignalsSeperatelyBool: Bool
    @param maxValue: The maximum value in mV which is shown.
    @type maxValue: Float
    @param meanBool: If True, the average of signals corresponding to a group is shown.
    @type meanBool: Bool
    """
    # Checks
    electrodes = findElectrodeIndices(electrodesUsedForTraining, electrodesToPlot)
    electrodesUsedForTraining = electrodesUsedForTraining.astype(np.int)
    del electrodesToPlot
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    groundTruth = groundTruth.ravel()
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            raise Exception("Shape of predictions and ground truths do not coincide.")
    if groundTruth.shape[0] != inputSignals.shape[0]:
        raise Exception("Number of ground truths does not coincide with number of samples.")
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")
    binaryLabelsBool = False
    if np.all(np.logical_or(groundTruth == 0, groundTruth == 1)):
        binaryLabelsBool = True
        groundTruth = groundTruth.astype(np.int)
    if not plotSignalsSeperatelyBool and not binaryLabelsBool:
        raise Exception("Ground truth contains unforseen labels.")
    if prediction is not None and not plotSignalsSeperatelyBool:
        if not np.all(np.logical_or(prediction == 0, prediction == 1)):
            raise Exception("Predictions contains unforseen labels.")
        else:
            prediction = prediction.astype(np.int)
    if plotSignalsSeperatelyBool:
        if inputSignals.shape[0] > 100:
            print("Warning: You will generate a lot of plots.")
        if meanBool:
            print("plotSignalsSeperatelyBool overwrites meanBool.")
        meanBool = False

    # PlotGeneration
    linSpace = np.arange(1, 1 + 2*inputSignals.shape[1], 2)
    cmap = cm.get_cmap(colourMap)

    if prediction is not None and binaryLabelsBool:
        # Use this to get rid of predictions close to the truth in the plot
        threshIndices = np.where(np.absolute(prediction - groundTruth) >= misClasThresh)
        # Depending on ground truth and prediction, we give each sample a number
        predictionLabel = np.atleast_1d(2 * groundTruth + np.absolute(np.around(prediction) - groundTruth))
        nrCorrectlyPredicted = np.argwhere(predictionLabel % 2 == 0).shape[0]
        nrOfSamples = predictionLabel.shape[0]
        predictionLabel = predictionLabel.astype(np.int)[threshIndices]
        inputSignals = inputSignals[threshIndices]

        colour = cmap(np.arange(4) / 3)
        custom_lines = [Line2D([0], [0], color=cmap(np.arange(4) / 3)[0], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[1], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[2], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[3], lw=2)]
    elif binaryLabelsBool:
        predictionLabel = groundTruth.astype(np.int)
        colour = cmap(np.array([0, 1, 2]) / 3)[[0, 2]]
        custom_lines = [Line2D([0], [0], color=cmap(np.arange(4))[0], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[2], lw=2), ]
    elif not binaryLabelsBool:
        predictionLabel = np.zeros(groundTruth.shape).astype(np.int)
        colour = np.atleast_2d(cmap(np.array([0, 1]))[1])
    else:
        raise Exception("Some unskilled monkeys were at work here... report that they get fired.")

    if meanBool:
        for e in electrodes:
            fig, ax = plt.subplots()
            if prediction is not None:
                ax.legend(custom_lines, ["0-0", "0-1", "1-1", "1-0"])
            else:
                ax.legend(custom_lines, ["0", "1"])
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d ms'))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d mv'))
            ax.set_ylim(bottom=-maxValue, top=maxValue)
            for i in range(colour.shape[0]):
                if (inputSignals[np.where(predictionLabel == i), :, :].ndim and
                    inputSignals[np.where(predictionLabel == i), :, :].size) != 0:
                    averageSignals = np.squeeze(np.mean(inputSignals[np.where(predictionLabel == i), :, e], axis=1))
                    deviationSignals = np.squeeze(np.std(inputSignals[np.where(predictionLabel == i), :, e], axis=1))
                    ax.plot(linSpace, averageSignals, c=colour[i], lw=1.5)
                    ax.fill_between(linSpace, averageSignals + deviationSignals, averageSignals - deviationSignals,
                                    facecolor=np.squeeze(colour[i]), alpha=0.1)
            if prediction is not None:
                ax.set_title("Acc.: {}%".format(round(nrCorrectlyPredicted / max(nrOfSamples, 1) * 100, 2)))
            if saveBool:
                fig.savefig(os.path.join(directory, filename) + "_El{}.{}".format(str(electrodesUsedForTraining[e]),format),
                            format=format, transparent=True)
            else:
                plt.show()
            plt.close()

    else:
        for e in electrodes:
            fig, ax = plt.subplots()
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d ms'))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d mv'))
            ax.set_ylim(bottom=-maxValue, top=maxValue)
            for i in range(inputSignals.shape[0]):
                if plotSignalsSeperatelyBool:
                    plt.close()
                    fig, ax = plt.subplots()
                    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d ms'))
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d mv'))
                    if prediction is not None:
                        ax.legend([Line2D([0], [0], color=colour[predictionLabel[i]], lw=2)],
                                  ["{} - {}".format(str(groundTruth[i]),str(prediction[i]))])
                    else:
                        ax.legend([Line2D([0], [0], color=colour[predictionLabel[i]], lw=2)],
                                  [str(groundTruth[i])])
                    ax.plot(linSpace, inputSignals[i, :, e], c=colour[predictionLabel[i]], lw=0.5)
                    ax.set_ylim(bottom=-maxValue, top=maxValue)
                    if saveBool:
                        fig.savefig(os.path.join(directory,filename) + "_Sample{}_El{}.{}".format(i,str(electrodesUsedForTraining[e]),format),
                                    format=format,transparent=True)
                    else:
                        plt.show()
                else:
                    if prediction is not None:
                        ax.legend(custom_lines, ["0-0", "0-1", "1-1", "1-0"])
                    else:
                        ax.legend(custom_lines, ["0", "1"])
                    ax.plot(linSpace, inputSignals[i, :, e], c=colour[predictionLabel[i]], lw=0.5)
            if not plotSignalsSeperatelyBool:
                if prediction is not None:
                    ax.set_title("Acc.: {}%".format(round(nrCorrectlyPredicted / max(nrOfSamples, 1) * 100, 2)))
                if saveBool:
                    fig.savefig(os.path.join(directory,filename) + "_El{}.{}".format(str(electrodesUsedForTraining[e]), format),
                                format=format, transparent=True)
                else:
                    plt.show()



def plotSignalAngle(inputSignals: np.ndarray, groundTruth: np.ndarray, directory: str, prediction: np.ndarray = None,
                    electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                    electrodesToPlot: np.ndarray = np.arange(1, 130),
                    colourMap: str = 'gist_rainbow', saveBool: bool = True,
                    filename: str = 'SignalVisualisation', format: str = 'pdf',
                    maxValue: float = 100, percentageThresh: float = 0, nrOfLevels: int = 4):

    """
    Visualises and colour codes the signals according to ground truth and prediction.

    @param inputSignals: 3d Tensor of the signals which have to be plotted. Shape has to be [#samples,#timestamps,
    #electrodes]
    @type inputSignals: Numpy Array
    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param electrodesUsedForTraining: Electrodes on which the network has been trained. These are the electrodes which
    correspond to the indices in the last dimension.
    @type electrodesUsedForTraining: Numpy Array
    @param electrodesToPlot: Electrodes for which the plots are to be generated.
    @type electrodesToPlot: Numpy Array
    @param colourMap: Matplotlib colour map for the plot.
    @type colourMap: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param maxValue: The maximum value in mV which is shown.
    @type maxValue: Float
    @param percentageThresh: Since signals are discretized into an arbitrary amount of levels, the legend may
    become cluttered. To avoid this, only show signals whos prediction correspond to atleast this percentage
    of the ground truth class.
    @type percentageThresh: Float
    @param nrOfLevels: Number, which defines in how many classes the regression problem is split. Has to be at least 2.
    @type nrOfLevels: Integer
    """

    # Checks
    electrodes = findElectrodeIndices(electrodesUsedForTraining, electrodesToPlot)
    electrodesUsedForTraining = electrodesUsedForTraining.astype(np.int)
    del electrodesToPlot
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    groundTruth = groundTruth.ravel()
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            raise Exception("Shape of predictions and ground truths do not coincide.")
    if nrOfLevels < 2:
        raise Exception("Need at least two levels.")
    if groundTruth.shape[0] != inputSignals.shape[0]:
        raise Exception("Number of ground truths does not coincide with number of samples.")
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")

    cmap = cm.get_cmap(colourMap)
    colour = cmap(np.arange(nrOfLevels) / (nrOfLevels - 1))

    def angleError(target,pred):
        return np.absolute(np.arctan2(np.sin(target-pred), np.cos(target-pred)))

    centers = np.linspace(-np.pi, np.pi, nrOfLevels + 1)[:-1]
    if prediction is not None:
        error = np.sqrt(np.mean(np.square(angleError(groundTruth,prediction))))
        distances = np.zeros([prediction.shape[0]]) + np.Infinity
        closestCenter = np.zeros([prediction.shape[0]], dtype=np.int)
    closestCenterTruth = np.zeros([groundTruth.shape[0]], dtype=np.int)
    distancesTruth = np.zeros([groundTruth.shape[0]]) + np.Infinity

    # Clustering
    for j in range(centers.shape[0]):
        if prediction is not None:
            distance = angleError(centers[j],prediction)
            closestCenter[np.where(np.squeeze(distance - distances) < 0)] = j
            distances = np.minimum(distances, distance)

        distance = angleError(centers[j], groundTruth)
        closestCenterTruth[np.where(np.squeeze(distance - distancesTruth) < 0)] = j
        distancesTruth = np.minimum(distancesTruth, distance)

    linSpace = np.arange(1, 2 * inputSignals.shape[1] + 1, 2)

    fig, ax = plt.subplots(figsize=(2 * centers.shape[0] + 2, 2 * centers.shape[0] + 2),
                           dpi=160 / max(2,np.log2(nrOfLevels)))
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    if prediction is not None:
        plt.title("Avg Angle Error: {}".format(round(error * 100, 1)), loc='left')
    angleDistance = (centers[1] - centers[0]) / 2
    for e in electrodes:
        for j in range(centers.shape[0]):
            plt.plot(np.array([0, np.cos(centers[j] + angleDistance)]),
                     np.array([0, np.sin(centers[j] + angleDistance)]), c='black', alpha=0.2)
            axes = ax.inset_axes([0.4 * np.cos(centers[j]) + 0.5 - 0.75 / max(6,nrOfLevels),
                                  np.absolute(1 - 0.4 * np.sin(centers[j]) - 0.5 - 0.75 / max(6,nrOfLevels)), 1.5 / max(4+nrOfLevels/2,nrOfLevels),
                                  1.5 / max(4+nrOfLevels/2,nrOfLevels)])
            axes.locator_params(nbins=3)
            axes.get_xaxis().set_major_formatter(FormatStrFormatter('%d ms'))
            axes.get_yaxis().set_major_formatter(FormatStrFormatter('%d mv'))
            meanTruthSignal = np.squeeze(np.mean(inputSignals[np.where(closestCenterTruth == j), :, e], axis=1))
            nrOfTruths = inputSignals[np.where(closestCenterTruth == j), :, e].shape[1]
            nrCorrectlyPredicted = 0
            deviationTruthSignal = np.squeeze(np.std(inputSignals[np.where(closestCenterTruth == j), :, e], axis=1))
            axes.plot(linSpace, meanTruthSignal, c=np.squeeze(colour[j]), lw=1.5, label=str(j) + ",({})".format(nrOfTruths))
            axes.fill_between(linSpace, meanTruthSignal + deviationTruthSignal, meanTruthSignal - deviationTruthSignal,
                              facecolor=np.squeeze(colour[j]), alpha=0.1)
            if prediction is not None:
                for n in range(centers.shape[0]):
                    if np.logical_and(closestCenterTruth == j, closestCenter == n).any():
                        nrPredicted = \
                        inputSignals[np.where(np.logical_and(closestCenterTruth == j, closestCenter == n)), :, e].shape[1]
                        if n == j:
                            nrCorrectlyPredicted = nrPredicted
                        if 100 * nrPredicted / nrOfTruths >= percentageThresh:
                            meanSignal = np.squeeze(np.mean(
                                inputSignals[np.where(np.logical_and(closestCenterTruth == j, closestCenter == n)), :,
                                e], axis=1))
                            deviationSignal = np.squeeze(np.std(
                                inputSignals[np.where(np.logical_and(closestCenterTruth == j, closestCenter == n)), :,
                                e], axis=1))
                            axes.plot(linSpace, meanSignal, linestyle='--', c=colour[n], lw=1,
                                      label="{}-{},({}%)".format(j, n, round(100 * nrPredicted / nrOfTruths)))
                            axes.fill_between(linSpace, meanSignal + deviationSignal, meanSignal - deviationSignal,
                                              facecolor=np.squeeze(colour[n]), alpha=0.1)
                axes.title.set_text("Coord: {}°, Acc.: {}%".format(round(centers[j] / np.pi * 180, 1),
                                                                   round(nrCorrectlyPredicted / max(nrOfTruths, 1) * 100,
                                                                         1)))

            axes.set_ylim(bottom=-maxValue, top=maxValue)
            axes.legend()
        if saveBool:
            fig.savefig(os.path.join(directory ,filename) + "_El{}.{}".format(str(electrodesUsedForTraining[e]), format),
                        format=format, transparent=True)
        else:
            plt.show()
        plt.close()


def plotSignalAmplitude(inputSignals: np.ndarray, groundTruth: np.ndarray, directory: str,
                        prediction: np.ndarray = None,
                        electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                        electrodesToPlot: np.ndarray = np.arange(1, 130),
                        colourMap: str = 'gist_rainbow', saveBool: bool = True,
                        filename: str = 'SignalVisualisation', format: str = 'pdf',
                        maxValue: float = 100, percentageThresh: float = 0, nrOfLevels: int = 4,
                        maxDistance = 800):

    """
    Visualises and colour codes the signals according to ground truth and prediction.

    @param inputSignals: 3d Tensor of the signals which have to be plotted. Shape has to be [#samples,#timestamps,
    #electrodes]
    @type inputSignals: Numpy Array
    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param electrodesUsedForTraining: Electrodes on which the network has been trained. These are the electrodes which
    correspond to the indices in the last dimension.
    @type electrodesUsedForTraining: Numpy Array
    @param electrodesToPlot: Electrodes for which the plots are to be generated.
    @type electrodesToPlot: Numpy Array
    @param colourMap: Matplotlib colour map for the plot.
    @type colourMap: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param maxValue: The maximum value in mV which is shown.
    @type maxValue: Float
    @param percentageThresh: Since signals are discretized into an arbitrary amount of levels, the legend may
    become cluttered. To avoid this, only show signals whos prediction correspond to atleast this percentage
    of the ground truth class.
    @type percentageThresh: Float
    @param nrOfLevels: Number, which defines in how many classes the regression problem is split. Has to be at least 2.
    @type nrOfLevels: Integer
    @param maxDistance: The maximal amplitude which is to be expected.
    @type maxDistance: Float
    """

    # Checks
    electrodes = findElectrodeIndices(electrodesUsedForTraining, electrodesToPlot)
    electrodesUsedForTraining = electrodesUsedForTraining.astype(np.int)
    del electrodesToPlot
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    groundTruth = groundTruth.ravel()
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            raise Exception("Shape of predictions and ground truths do not coincide.")
    if nrOfLevels < 2:
        raise Exception("Need at least two levels.")
    if groundTruth.shape[0] != inputSignals.shape[0]:
        raise Exception("Number of ground truths does not coincide with number of samples.")
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")

    #Plot Generation
    cmap = cm.get_cmap(colourMap)
    colour = cmap(np.arange(nrOfLevels) / (nrOfLevels - 1))
    centers = np.linspace(0, maxDistance, nrOfLevels + 1)[:-1]
    centers += (centers[1] - centers[0]) / 2
    if prediction is not None:
        error = np.mean(np.absolute(groundTruth-prediction))
        distances = np.zeros((prediction.shape[0])) + np.Infinity
        closestCenter = np.zeros([prediction.shape[0]], dtype=np.int)

    closestCenterTruth = np.zeros([groundTruth.shape[0]], dtype=np.int)
    distancesTruth = np.zeros((groundTruth.shape[0])) + np.Infinity

    # Clustering
    for j in range(centers.shape[0]):
        if prediction is not None:
            distance = np.absolute(prediction - centers[j])
            closestCenter[np.where(np.squeeze(distance - distances) < 0)] = j
            distances = np.minimum(distances, distance)

        distance = np.absolute(groundTruth - centers[j])
        closestCenterTruth[np.where(np.squeeze(distance - distancesTruth) < 0)] = j
        distancesTruth = np.minimum(distancesTruth, distance)

    linSpace = np.arange(1, 2*inputSignals.shape[1]+1, 2)

    for e in electrodes:
        fig, ax = plt.subplots(figsize=(8, 4 * centers.shape[0]), dpi=160 / np.log2(nrOfLevels))
        ax.get_xaxis().set_visible(False)
        ax.set_ylim([0, maxDistance])
        ax.set_ylabel('px')
        # centers[1] - centers[0] is used to find the general distance between to center points
        ax.set_yticks(centers - (centers[1] - centers[0]) / 2)
        if prediction is not None:
            ax.title.set_text("Average Error Distance: {} Pixels".format(error))
        plt.grid()
        for j in range(centers.shape[0]):
            axes = ax.inset_axes([0.15, (0.1 + j) / nrOfLevels, 0.8, 0.8 / nrOfLevels])
            axes.get_xaxis().set_major_formatter(FormatStrFormatter('%d ms'))
            axes.get_yaxis().set_major_formatter(FormatStrFormatter('%d mv'))
            meanTruthSignal = np.squeeze(np.mean(inputSignals[np.where(closestCenterTruth == j), :, e], axis=1))
            nrOfTruths = inputSignals[np.where(closestCenterTruth == j), :, e].shape[1]
            nrCorrectlyPredicted = 0
            deviationTruthSignal = np.squeeze(np.std(inputSignals[np.where(closestCenterTruth == j), :, e], axis=1))
            axes.plot(linSpace, meanTruthSignal, c=colour[j], lw=1.5,
                      label=str(j) + ",({})".format(nrOfTruths))
            axes.fill_between(linSpace, meanTruthSignal + deviationTruthSignal, meanTruthSignal - deviationTruthSignal,
                              facecolor=np.squeeze(colour[j]), alpha=0.1)
            if prediction is not None:
                for n in range(centers.shape[0]):
                    if np.logical_and(closestCenterTruth == j, closestCenter == n).any():
                        nrPredicted = \
                        inputSignals[np.where(np.logical_and(closestCenterTruth == j, closestCenter == n)), :,
                        e].shape[1]
                        if n == j:
                            nrCorrectlyPredicted = nrPredicted
                        if 100 * nrPredicted / nrOfTruths >= percentageThresh:
                            meanSignal = np.squeeze(np.mean(
                                inputSignals[np.where(np.logical_and(closestCenterTruth == j, closestCenter == n)),
                                :, e], axis=1))
                            deviationSignal = np.squeeze(np.std(
                                inputSignals[np.where(np.logical_and(closestCenterTruth == j, closestCenter == n)),
                                :, e], axis=1))
                            axes.plot(linSpace, meanSignal, linestyle='--', c=np.squeeze(colour[n]), lw=1,
                                      label="{}-{},({}%)".format(j, n, round(100 * nrPredicted / nrOfTruths)))
                            axes.fill_between(linSpace, meanSignal + deviationSignal, meanSignal - deviationSignal,
                                              facecolor=np.squeeze(colour[n]), alpha=0.1)
                axes.title.set_text("Coord: {}, Acc.: {}%".format(np.around(centers[j]),
                                                                  round(nrCorrectlyPredicted / max(nrOfTruths, 1) * 100,
                                                                        1)))
            axes.set_ylim(bottom=-maxValue, top=maxValue)
            axes.legend()
        if saveBool:
            fig.savefig(os.path.join(directory ,filename) + "_El{}.{}".format(str(electrodesUsedForTraining[e]), format),
                        format=format, transparent=True)
        else:
            plt.show()
        plt.close()


def plotSignalPosition(inputSignals: np.ndarray, groundTruth: np.ndarray, directory: str, prediction: np.ndarray = None,
                       electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                       electrodesToPlot: np.ndarray = np.arange(1, 130),
                       colourMap: str = 'gist_rainbow', filename: str = 'SignalVisualisation', format: str = 'pdf',
                       maxValue: float = 100, percentageThresh: float = 0, nrOfLevels: int = 4,
                       BoundariesX: (int,int) = (0,800), BoundariesY: (int,int) = (0,600), saveBool: bool = True):

    """
    Visualises and colour codes the signals according to ground truth and prediction.

    @param inputSignals: 3d Tensor of the signals which have to be plotted. Shape has to be [#samples,#timestamps,
    #electrodes]
    @type inputSignals: Numpy Array
    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param electrodesUsedForTraining: Electrodes on which the network has been trained. These are the electrodes which
    correspond to the indices in the last dimension.
    @type electrodesUsedForTraining: Numpy Array
    @param electrodesToPlot: Electrodes for which the plots are to be generated.
    @type electrodesToPlot: Numpy Array
    @param colourMap: Matplotlib colour map for the plot.
    @type colourMap: String
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param maxValue: The maximum value in mV which is shown.
    @type maxValue: Float
    @param percentageThresh: Since signals are discretized into an arbitrary amount of levels, the legend may
    become cluttered. To avoid this, only show signals whos prediction correspond to atleast this percentage
    of the ground truth class.
    @type percentageThresh: Float
    @param nrOfLevels: Number, which defines in how many classes the regression problem is split. Has to be at least 2.
    @type nrOfLevels: Integer
    @param BoundariesX: Tuple of Integers, which determines the lower and upper bound of the area of the x-axis of the plot.
    @type BoundariesX: (Integer,Integer)
    @param BoundariesY: Tuple of Integers, which determines the lower and upper bound of the area of the y-axis of the plot.
    @type BoundariesY: (Integer,Integer)
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    """

    # Checks
    electrodes = findElectrodeIndices(electrodesUsedForTraining, electrodesToPlot)
    electrodesUsedForTraining = electrodesUsedForTraining.astype(np.int)
    del electrodesToPlot
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    if groundTruth.ndim != 2:
        raise Exception("Need a 2 dimensional array as ground truth.")
    if groundTruth.shape[1] != 2:
        raise Exception("Second dimension must be of size 2 of the ground truth array.")
    if prediction is not None:
        if groundTruth.shape[0] != prediction.shape[0] or groundTruth.shape[1] != prediction.shape[1]:
            raise Exception("Shape of predictions and ground truths do not coincide.")
    if groundTruth.shape[0] != inputSignals.shape[0]:
        raise Exception("Number of ground truths does not coincide with number of samples.")
    if nrOfLevels < 2:
        raise Exception("Need at least two levels.")
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")

    cmap = cm.get_cmap(colourMap)
    colour = cmap(np.arange(nrOfLevels) / (nrOfLevels - 1))

    #The goal here is to factorize an integer in two large integers.
    gridsX = 1
    gridsY = nrOfLevels
    loss = abs(gridsX - gridsY)
    while (gridsX != nrOfLevels):
        if (nrOfLevels / gridsX) % 1 == 0 and loss > abs(gridsX - (nrOfLevels / gridsX)):
            gridsY = int(nrOfLevels / gridsX)
            loss = abs(gridsX - gridsY)
        gridsX += 1
    gridsX = int(nrOfLevels / gridsY)
    centers = np.zeros([gridsX, gridsY, 2])

    # Determine the center points. A Position then is clustered to the nearest center.
    # Centers will be labeled with an integer, starting count from top left and rowwise.
    centers[:, :, 0] = np.expand_dims(
        np.arange(centers.shape[0]) * (BoundariesX[1] - BoundariesX[0]) / centers.shape[0] +
        (BoundariesX[1] - BoundariesX[0]) / (2 * centers.shape[0]), axis=1) + BoundariesX[0]
    centers[:, :, 1] = np.tile((np.arange(centers.shape[1]) * (BoundariesY[1] - BoundariesY[0]) / centers.shape[1] + (
                BoundariesY[1] - BoundariesY[0]) / (2 * centers.shape[1]))[::-1], (centers.shape[0], 1)) + BoundariesY[0]

    if prediction is not None:
        error = np.sqrt(np.linalg.norm(groundTruth - prediction, axis=1).mean())
        distances = np.zeros(prediction.shape[0]) + np.Infinity
        closestCenter = np.zeros(prediction.shape[0], dtype=np.int)
    closestCenterTruth = np.zeros(groundTruth.shape[0], dtype=np.int)
    distancesTruth = np.zeros(groundTruth.shape[0]) + np.Infinity

    #Clustering
    for j in range(centers.shape[1]):
        for i in range(centers.shape[0]):
            if prediction is not None:
                distance = np.power(prediction[:, 0] - centers[i, j, 0], 2) + np.power(prediction[:, 1] - centers[i, j, 1],2)
                closestCenter[np.where(np.squeeze(distance - distances) < 0)] = i + j * centers.shape[0]
                distances = np.minimum(distances, distance)

            distance = np.power(groundTruth[:, 0] - centers[i, j, 0], 2) + np.power(groundTruth[:, 1] - centers[i, j, 1], 2)
            closestCenterTruth[np.where(np.squeeze(distance - distancesTruth) < 0)] = i + j * centers.shape[0]
            distancesTruth = np.minimum(distancesTruth, distance)

    linSpace = np.arange(1,2*inputSignals.shape[1]+1,2)

    for e in electrodes:
        fig, ax = plt.subplots(figsize=(8 * centers.shape[0], 4 * centers.shape[1]), dpi=160 / np.log2(nrOfLevels))
        ax.set_xlim([BoundariesX[0], BoundariesX[1]])
        ax.set_ylim([BoundariesY[0], BoundariesY[1]])
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, centers.shape[0] + 1) * (BoundariesX[1] - BoundariesX[0]) / centers.shape[0] + BoundariesX[0])
        ax.set_yticks(np.arange(0, centers.shape[1] + 1) * (BoundariesY[1] - BoundariesY[0]) / centers.shape[1] + BoundariesY[0])
        ax.set_xlabel('px')
        ax.set_ylabel('px')
        if prediction is not None:
            ax.title.set_text("Average Error Distance: {}px".format(error))
        plt.grid()
        for i in range(centers.shape[1]):
            for j in range(centers.shape[0]):
                axes = ax.inset_axes(
                    [(centers[j, i, 0] - BoundariesX[0]) / (BoundariesX[1] - BoundariesX[0]) - 0.7 / (2 * centers.shape[0]),
                     np.absolute(
                         1 - (centers[j, i, 1] - BoundariesY[0]) / (BoundariesY[1] - BoundariesY[0]) - 0.8 / (2 * centers.shape[1])),
                     0.8 / centers.shape[0], 0.8 / centers.shape[1]])
                index = j + i * centers.shape[0]
                axes.get_xaxis().set_major_formatter(FormatStrFormatter('%d ms'))
                axes.get_yaxis().set_major_formatter(FormatStrFormatter('%d mv'))
                groundTruthSignal = np.squeeze(np.mean(inputSignals[np.where(closestCenterTruth == index), :, e], axis=1))
                nrOfTruths = inputSignals[np.where(closestCenterTruth == index), :, e].shape[1]
                nrCorrectlyPredicted = 0
                deviationTruthSignal = np.squeeze(np.std(inputSignals[np.where(closestCenterTruth == index), :, e], axis=1))
                axes.plot(linSpace, groundTruthSignal, c=colour[index], lw=1.5,
                          label=str(index) + ",({})".format(nrOfTruths))
                axes.fill_between(linSpace, groundTruthSignal + deviationTruthSignal,
                                  groundTruthSignal - deviationTruthSignal, facecolor=np.squeeze(colour[index]),
                                  alpha=0.1)
                if prediction is not None:
                    for n in range(centers.shape[0] * centers.shape[1]):
                        if np.logical_and(closestCenterTruth == index, closestCenter == n).any():
                            nrPredicted = \
                            inputSignals[np.where(np.logical_and(closestCenterTruth == index, closestCenter == n)), :, e].shape[1]
                            if n == index:
                                nrCorrectlyPredicted = nrPredicted
                            if 100 * nrPredicted / nrOfTruths >= percentageThresh:
                                meanSignal = np.squeeze(np.mean(
                                    inputSignals[np.where(np.logical_and(closestCenterTruth == index, closestCenter == n)), :, e],
                                    axis=1))
                                deviationSignal = np.squeeze(np.std(
                                    inputSignals[np.where(np.logical_and(closestCenterTruth == index, closestCenter == n)), :, e],
                                    axis=1))
                                axes.plot(linSpace, np.squeeze(meanSignal), linestyle='--', c=np.squeeze(colour[n]), lw=1,
                                          label="{}-{},({}%)".format(index, n, round(100 * nrPredicted / nrOfTruths)))
                                axes.fill_between(linSpace, meanSignal + deviationSignal, meanSignal - deviationSignal,
                                                  facecolor=np.squeeze(colour[n]), alpha=0.1)
                    axes.title.set_text(
                        "Coord:[{},{}], Acc.: {}%".format(np.around(centers[j, i, 0]), np.around(centers[j, i, 1]),
                                                          round(nrCorrectlyPredicted / max(nrOfTruths, 1) * 100, 1)))
                axes.set_ylim(bottom=-maxValue, top=maxValue)
                axes.legend()

        if saveBool:
            fig.savefig(os.path.join(directory, filename) + "_El{}.{}".format(str(electrodesUsedForTraining[e]), format), format=format,
                        transparent=True)
        else:
            plt.show()
        plt.close()


def plotEyeLR(eyeMovement: np.ndarray, groundTruth: np.ndarray, directory: str, prediction: np.ndarray = None,
                 colourMap: str = 'gist_rainbow', misClasThresh: float = 0,
                 filename: str = 'SignalVisualisation', format: str = 'pdf', saveBool: bool = True,
                 plotSignalsSeperatelyBool: bool = False, maxValue: int = 800, meanBool: bool = True):

    """
    Visualises and colour codes the eye movements according to ground truth and prediction.

    @param eyeMovement: 2d Tensor of the eye movements which have to be plotted. Shape has to be [#samples,#timestamps]
    @type eyeMovement: Numpy Array
    @param groundTruth: Corresponding ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param prediction: Predicted label for each sample. If None, the plot is based only on the ground truth.
    @type prediction: Numpy Array
    @param colourMap: Matplotlib colour map for the plot.
    @type colourMap: String
    @param misClasThresh: Only plots signals which the difference between prediction and ground truth is larger
    than this.
    @type misClasThresh: Float
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param plotSignalsSeperatelyBool: If True, each signal is plotted in a new window.
    @type plotSignalsSeperatelyBool: Bool
    @param maxValue: The maximum value in px which is shown.
    @type maxValue: Integer
    @param meanBool: If True, the average of signals corresponding to a group is shown.
    @type meanBool: Bool
    """
    # Checks
    if eyeMovement.ndim != 2:
        raise Exception("Need a 2 dimensional array as input.")
    groundTruth = groundTruth.ravel()
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            raise Exception("Shape of predictions and ground truths do not coincide.")
    if groundTruth.shape[0] != eyeMovement.shape[0]:
        raise Exception("Number of ground truths does not coincide with number of samples.")
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")
    binaryLabelsBool = True
    if not np.all(np.logical_or(groundTruth == 0, groundTruth == 1)):
        binaryLabelsBool = False
    if not plotSignalsSeperatelyBool and not binaryLabelsBool:
        raise Exception("Ground truth contains unforseen labels.")
    if prediction is not None and not plotSignalsSeperatelyBool:
        if not np.all(np.logical_or(prediction == 0, prediction == 1)):
            raise Exception("Predictions contains unforseen labels.")
    if plotSignalsSeperatelyBool:
        if eyeMovement.shape[0] > 100:
            print("Warning: You will generate a lot of plots.")
        if meanBool:
            print("plotSignalsSeperatelyBool overwrites meanBool.")
        meanBool = False

    # PlotGeneration
    linSpace = np.arange(1, 1 + 2*eyeMovement.shape[1], 2)
    cmap = cm.get_cmap(colourMap)

    if prediction is not None and binaryLabelsBool:
        # Use this to get rid of predictions close to the truth in the plot
        threshIndices = np.where(np.absolute(prediction - groundTruth) >= misClasThresh)
        # Depending on ground truth and prediction, we give each sample a number
        predictionLabel = np.atleast_1d(2 * groundTruth + np.absolute(np.around(prediction) - groundTruth))
        nrCorrectlyPredicted = np.argwhere(predictionLabel % 2 == 0).shape[0]
        nrOfSamples = predictionLabel.shape[0]
        predictionLabel = predictionLabel.astype(np.int)[threshIndices]
        eyeMovement = eyeMovement[threshIndices]

        colour = cmap(np.arange(4) / 3)
        custom_lines = [Line2D([0], [0], color=cmap(np.arange(4) / 3)[0], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[1], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[2], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[3], lw=2)]
    elif binaryLabelsBool:
        predictionLabel = groundTruth.astype(np.int)
        colour = cmap(np.array([0, 1, 2]) / 3)[[0, 2]]
        custom_lines = [Line2D([0], [0], color=cmap(np.arange(4))[0], lw=2),
                        Line2D([0], [0], color=cmap(np.arange(4) / 3)[2], lw=2), ]
    elif not binaryLabelsBool:
        predictionLabel = np.zeros(groundTruth.shape).astype(np.int)
        colour = np.atleast_2d(cmap(np.array([0, 1]))[1])
    else:
        raise Exception("Some unskilled monkeys were at work here... report that they get fired.")

    if meanBool:
        fig, ax = plt.subplots()
        if prediction is not None:
            ax.legend(custom_lines, ["0-0", "0-1", "1-1", "1-0"])
        else:
            ax.legend(custom_lines, ["0", "1"])
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d px'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d ms'))
        ax.set_xlim(left=0, right=maxValue)
        for i in range(colour.shape[0]):
            if (eyeMovement[np.where(predictionLabel == i), :].ndim and
                eyeMovement[np.where(predictionLabel == i), :].size) != 0:
                averageSignals = np.squeeze(np.mean(eyeMovement[np.where(predictionLabel == i), :], axis=1))
                deviationSignals = np.squeeze(np.std(eyeMovement[np.where(predictionLabel == i), :], axis=1))
                ax.plot(averageSignals, linSpace,  c=colour[i], lw=1.5)
                ax.fill_betweenx(linSpace, averageSignals + deviationSignals, averageSignals - deviationSignals,
                                facecolor=np.squeeze(colour[i]), alpha=0.1)
        if prediction is not None:
            ax.set_title("Acc.: {}%".format(round(nrCorrectlyPredicted / max(nrOfSamples, 1) * 100, 2)))
        if saveBool:
            fig.savefig(os.path.join(directory, filename) + "_Eye.{}".format(format), format=format, transparent=True)
        else:
            plt.show()
        plt.close()

    else:
        fig, ax = plt.subplots()
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d px'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d ms'))
        ax.set_xlim(left=0, right=maxValue)
        for i in range(eyeMovement.shape[0]):
            if plotSignalsSeperatelyBool:
                plt.close()
                fig, ax = plt.subplots()
                plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d px'))
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d ms'))
                if prediction is not None:
                    ax.legend([Line2D([0], [0], color=colour[predictionLabel[i]], lw=2)],
                              ["{} - {}".format(str(groundTruth[i]),str(prediction[i]))])
                else:
                    ax.legend([Line2D([0], [0], color=colour[predictionLabel[i]], lw=2)],
                              [str(groundTruth[i])])
                ax.plot(eyeMovement[i, :], linSpace, c=colour[predictionLabel[i]], lw=0.5)
                ax.set_xlim(left=0, right=maxValue)
                if saveBool:
                    fig.savefig(os.path.join(directory,filename) + "_Sample{}_Eye.{}".format(i,format),
                                format=format,transparent=True)
                else:
                    plt.show()
            else:
                if prediction is not None:
                    ax.legend(custom_lines, ["0-0", "0-1", "1-1", "1-0"])
                else:
                    ax.legend(custom_lines, ["0", "1"])
                ax.plot(eyeMovement[i, :], linSpace, c=colour[predictionLabel[i]], lw=0.5)
        if not plotSignalsSeperatelyBool:
            if prediction is not None:
                ax.set_title("Acc.: {}%".format(round(nrCorrectlyPredicted / max(nrOfSamples, 1) * 100, 2)))
            if saveBool:
                fig.savefig(os.path.join(directory,filename) + "_Eye.{}".format( format),
                            format=format, transparent=True)
            else:
                plt.show()




###############Helper Functions#####################

def findElectrodeIndices(electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                         electrodesToPlot: np.ndarray = np.arange(1, 130)) -> np.ndarray:
    intersect, ind_a, electrodes = np.intersect1d(electrodesToPlot, electrodesUsedForTraining, return_indices=True)
    del intersect, ind_a

    return np.atleast_1d(electrodes)