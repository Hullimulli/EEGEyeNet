import numpy as np

from utils import IOHelper
from config import config
from benchmark import split
from Joels_Files.plotFunctions import electrode_plots
from Joels_Files.plotFunctions import signal_plots
asdf = np.arange(129)
#electrode_plots.electrodeBarPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#colours = electrode_plots.colourCode(-asdf)
#electrode_plots.electrodePlot(colours,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")

#electrode_plots.topoPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
electrodeIndices = np.array([1,32])

#trainIndices, valIndices, testIndices = split(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:,0], 0.7, 0.15, 0.15)
trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:10]
trainX = trainX[:,:, electrodeIndices - 1]
trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:10,1]
prediction = np.array([0,0,0,1,1,1,0,0,0,1])
asdf = np.array([1])
asdf = asdf.ravel()
signal_plots.plotSignalLR(inputSignals=trainX,groundTruth=trainY,plotSignalsSeperatelyBool=False,meanBool=False,prediction=prediction,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesUsedForTraining=electrodeIndices,electrodesToPlot=electrodeIndices)