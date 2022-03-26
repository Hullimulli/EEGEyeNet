import numpy as np

from Joels_Files.plotFunctions import electrode_plots

asdf = np.zeros([129,2]) + 255
electrode_plots.electrodeBarPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
electrode_plots.electrodePlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
