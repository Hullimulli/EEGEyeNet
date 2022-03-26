import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
import os
import cv2
from pandas import read_csv
from pathlib import Path


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