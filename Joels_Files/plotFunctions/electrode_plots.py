import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like, Normalize
import os
import cv2
from pandas import read_csv
from pathlib import Path
import matplotlib.cm as cm
import scipy.io as sio
import mne

def electrodeBarPlot(values: np.ndarray , directory: str, yAxisLabel: str = "Loss Ratio",
                     filename: str = "Electrode_Loss" ,format: str = 'pdf' ,colour: str = 'red',
                     savePlotBool: bool = True):
    """
    Simple bar plot where each value is is visualised at its index + 1.
    @param values: Array of values, all values <0 are set to 0.
    @type values: Numpy Array
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param filename: Savename for the plot.
    @type filename: String
    @param format: File format for the save file.
    @type format: String
    @param colour: Colour for a matplotlib bar plot.
    @type colour: String
    @param savePlotBool: If True, saves the plot. Else it shows it.
    @type savePlotBool: Bool
    """

    #Checks
    values = np.atleast_1d(values).ravel()
    if not is_color_like(colour):
        print("Invalid colour. Using red.")
        colour = 'red'
    if not format in plt.gcf().canvas.get_supported_filetypes():
        print("Invalid format. Using pdf.")
        format = 'pdf'
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    #Generating Plot
    values[np.where(values < 0)] = 0
    xAxis = np.arange(values.shape[0]) + 1
    fig = plt.figure()
    plt.xlabel("Electrode Number")
    plt.ylabel(yAxisLabel)
    plt.bar(xAxis, values, color=colour)
    if savePlotBool:
        fig.savefig(os.path.join(directory,filename) + ".{}".format(format), format=format, transparent=True)
    else:
        plt.show()
    plt.close()


def electrodePlot(colourValues: np.ndarray, directory: str, filename: str = "Electrode_Configuration",
                  alpha: float = 0.4):
    """
    Shows colour coded circles representing electrode positions. The colours are specified by the user.
    @param colourValues: An array with the colour values which are used. Index of axis 0 is the electrode number - 1.
    @type colourValues: Numpy Array of shape [<=129,3]
    @param directory: Directory where the plot has to be saved.
    @type directory: String
    @param filename: Savename for the plot.
    @type filename: String
    @param alpha: Transparency of the colours.
    @type alpha: Float
    """
    #Checks
    colourValues = np.transpose(np.atleast_2d(np.transpose(colourValues)))
    if colourValues.shape[1] < 3:
        np.append(colourValues, np.zeros([colourValues.shape[0], 3 - colourValues.shape[1]]), axis=1)
    colourValues = colourValues[:129,:3]
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    #Generating Plot
    pathOfFile = os.path.join(Path(__file__).resolve().parent,"filesForPlot")
    img = cv2.imread(os.path.join(pathOfFile,'blank.png'), cv2.IMREAD_COLOR)
    overlay = img.copy()
    coord = read_csv(os.path.join(pathOfFile,'coord.csv'), index_col='electrode', dtype=int)
    for i in range(colourValues.shape[0]):
        pt = coord.loc[i+1]
        x, y, r = pt['posX'], pt['posY'], pt['radius']
        cv2.circle(overlay, (x, y), r, colourValues[i,:], -1)

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(directory,filename)+'.png', img)


def colourCode(values: np.ndarray, electrodes: np.ndarray = np.arange(1,130), colourMap: str = "Reds") -> np.ndarray:
    """
    Maps values to colours. Can be used for electrodePlot().
    @param values: Values which are translated to a colour map. If only one value is given, an array of length 129 is
    constructed.
    @type values: Numpy Array
    @param electrodes: Which indices+1 of the values are used for the colour map. Rest is mapped to white.
    @type electrodes: Numpy Array
    @param colourMap: Matplotlib colour map.
    @type colourMap: String
    @return: Numpy Array of shape [values.shape,3] containing the RGB values.
    @rtype: Numpy Array
    """

    #Checks
    values = np.atleast_1d(values)
    if values.shape[0] == 1:
        values = np.zeros(129) + values
    if colourMap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'Reds'.")
        colourMap = "Reds"
    if values.shape[0] < np.max(electrodes):
        print("Index corresponding to electrode {} does not exist. Returning all white array.".format(np.array2string(np.max(electrodes))))
        return np.zeros(values.shape[0],3) + 255

    #Mapping Values to Colours
    cmap = cm.get_cmap(colourMap)
    minVal = np.min(values[electrodes-1])
    maxVal = np.max(values[electrodes-1])
    if minVal == maxVal:
        minVal = min(0,minVal)
        maxVal = max(0,maxVal)
    norm = Normalize(vmin=minVal, vmax=maxVal)
    colours = cmap(norm(values))[:,0:3]
    setToWhiteMask = np.ones(values.shape, np.bool)
    setToWhiteMask[electrodes-1] = 0
    colours[setToWhiteMask] = np.array([1,1,1])
    colours[:,[2, 0]] = colours[:,[0, 2]]
    return colours * 255


def topoPlot(values: np.ndarray, directory: str, filename: str = 'topoPlot', format: str = 'pdf',
             figSize: (float, float) = (7,4.5), saveBool: bool = True, cmap: str = 'Reds',
             valueType: str = "Loss-Ratio", cutSmallerThanZeroBool: bool = True, epsilon: float = 0.01):
    """
    Generates a topographic map of an EEG field based on the input values.
    @param values: Array of length 129, where the index + 1 equals the electrode number and a value, which will be
    colour coded.
    @type values: Numpy Array
    @param filename: Name of the file as which the plot will be saved.
    @type filename: String
    @param format: Format of the save file.
    @type format: String
    @param figSize: Width and height in inches of the plot.
    @type figSize: (float,float)
    @param saveBool: If True, the plot will be saved. Else it will be shown.
    @type saveBool: Bool
    @param cmap: Matplotlib colourmap
    @type cmap: String
    @param epsilon: Number to adjust weighting in the log plot. Has to be larger than 0.
    @type epsilon: float
    @param valueType: What value is visualised.
    @type valueType: String
    @param cutSmallerThanZeroBool: If True, all elements of values smaller than zero are set to zero.
    @type cutSmallerThanZeroBool: Bool
    """


    #Checks
    if cmap not in plt.colormaps():
        print("Colourmap does not exist in Matplotlib. Using 'Reds'.")
        cmap = "Reds"
    if epsilon <= 0:
        print("Epsilon too small, using epsilon = 1.")
        epsilon = 1
    if values.shape[0] != 129:
        print("Wrong array dimensions.")
        return

    #Generating Plots
    pathOfFile = os.path.join(Path(__file__).resolve().parent, "filesForPlot")
    electrodePositions = sio.loadmat(os.path.join(pathOfFile,"lay129_head.mat"))['lay129_head']['pos'][0][0]
    outline = sio.loadmat(os.path.join(pathOfFile,"lay129_head.mat"))['lay129_head']['outline'][0][0]
    mask = sio.loadmat(os.path.join(pathOfFile,"lay129_head.mat"))['lay129_head']['mask'][0][0]
    if cutSmallerThanZeroBool:
        values[np.where(values < 0)] = 0
    else:
        values -= np.min(values)
    values = 10 * np.log(values + epsilon)
    fig = plt.figure(figsize=figSize)
    #Generating outline dictionary for mne topoplot
    outlines = dict()
    outlines["mask_pos"] = (mask[0,0][:,0],mask[0,0][:,1])
    outlines["head"] = (outline[0, 0][:,0],outline[0, 0][:,1])
    outlines["nose"] = (outline[0, 1][:,0],outline[0, 1][:,1])
    outlines["ear_left"] = (outline[0, 2][:,0],outline[0, 2][:,1])
    outlines["ear_right"] = (outline[0, 3][:,0],outline[0, 3][:,1])
    #This cuts out parts of the colour circle
    outlines['clip_radius'] = (0.5,) * 2
    outlines['clip_origin'] = (0,0.07)
    im, cm = mne.viz.plot_topomap(np.squeeze(values),electrodePositions[3:132,:],outlines=outlines,show=False,cmap=cmap)
    clb = fig.colorbar(im)
    if epsilon==1:
        clb.ax.set_title("{} in Db".format(valueType))
    else:
        clb.ax.set_title("10x Log of {}, eps={}".format(valueType,epsilon))
    if saveBool:
        fig.savefig(os.path.join(directory, filename) + ".{}".format(format), format=format, transparent=True)
    else:
        plt.show()
    plt.close()

