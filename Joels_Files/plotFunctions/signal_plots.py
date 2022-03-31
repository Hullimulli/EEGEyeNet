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

    @param inputSignals: 3d Tensor of the signal which have to be plotted. Shape has to be [#samples,#timestamps,
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

    @param eyeMovement: 3d Tensor of the signal which have to be plotted. Shape has to be [#samples,#timestamps]
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

    @param inputSignals: 3d Tensor of the signal which have to be plotted. Shape has to be [#samples,#timestamps,
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
        print("Need a 3 dimensional array as input.")
        return
    groundTruth = groundTruth.ravel()
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            print("Shape of predictions and ground truths do not coincide.")
            return
    if groundTruth.shape[0] != inputSignals.shape[0]:
        print("Number of ground truths does not coincide with number of samples.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return
    binaryLabelsBool = False
    if np.all(np.logical_or(groundTruth == 0, groundTruth == 1)):
        binaryLabelsBool = True
        groundTruth = groundTruth.astype(np.int)
    if not plotSignalsSeperatelyBool and not binaryLabelsBool:
        print("Ground truth contains unforseen labels.")
        return
    if prediction is not None and not plotSignalsSeperatelyBool:
        if not np.all(np.logical_or(prediction == 0, prediction == 1)):
            print("Predictions contains unforseen labels.")
            return
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
        print("Some unskilled monkeys were at work here... report that they get fired.")
        return

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
                    colourMap: str = 'gist_rainbow', nrOfSignals: int = 20000,
                    filename: str = 'SignalVisualisation', format: str = 'pdf',
                    maxValue: float = 1000, percentageThresh: float = 0, nrOfLevels: int = 4):
    pass
    # TODO


def plotSignalAmplitude(inputSignals: np.ndarray, groundTruth: np.ndarray, directory: str,
                        prediction: np.ndarray = None,
                        electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                        electrodesToPlot: np.ndarray = np.arange(1, 130),
                        colourMap: str = 'gist_rainbow', nrOfSignals: int = 20000,
                        filename: str = 'SignalVisualisation', format: str = 'pdf',
                        maxValue: float = 1000, percentageThresh: float = 0, nrOfLevels: int = 4):
    pass
    # TODO


def plotSignalPosition(inputSignals: np.ndarray, groundTruth: np.ndarray, directory: str, prediction: np.ndarray = None,
                       electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                       electrodesToPlot: np.ndarray = np.arange(1, 130),
                       colourMap: str = 'gist_rainbow', nrOfSignals: int = 20000,
                       filename: str = 'SignalVisualisation', format: str = 'pdf',
                       maxValue: float = 1000, percentageThresh: float = 0, nrOfLevels: int = 4):

    # Checks
    electrodes = findElectrodeIndices(electrodesUsedForTraining, electrodesToPlot)
    electrodesUsedForTraining = electrodesUsedForTraining.astype(np.int)
    del electrodesToPlot
    if inputSignals.ndim != 3:
        print("Need a 3 dimensional array as input.")
        return
    groundTruth = groundTruth.ravel().astype(np.int)
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            print("Shape of predictions and ground truths do not coincide.")
            return
    if groundTruth.shape[0] != inputSignals.shape[0]:
        print("Number of ground truths does not coincide with number of samples.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    # TODO



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
        print("Need a 2 dimensional array as input.")
        return
    groundTruth = groundTruth.ravel()
    if prediction is not None:
        prediction = prediction.ravel()
        if groundTruth.shape[0] != prediction.shape[0]:
            print("Shape of predictions and ground truths do not coincide.")
            return
    if groundTruth.shape[0] != eyeMovement.shape[0]:
        print("Number of ground truths does not coincide with number of samples.")
        return
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'gist_rainbow'.")
        colourMap = "gist_rainbow"
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return
    binaryLabelsBool = True
    if not np.all(np.logical_or(groundTruth == 0, groundTruth == 1)):
        binaryLabelsBool = False
    if not plotSignalsSeperatelyBool and not binaryLabelsBool:
        print("Ground truth contains unforseen labels.")
        return
    if prediction is not None and not plotSignalsSeperatelyBool:
        if not np.all(np.logical_or(prediction == 0, prediction == 1)):
            print("Predictions contains unforseen labels.")
            return
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
        print("Some unskilled monkeys were at work here... report that they get fired.")
        return

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