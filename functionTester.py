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
electrodeIndices = np.array([32])

#trainIndices, valIndices, testIndices = split(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:5000,0], 0.7, 0.15, 0.15)
trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:300,:, electrodeIndices - 1]
#eye = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:5000,:, 128]
trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:300,2]
for i in np.arange(2,17):
    signal_plots.plotSignalAngle(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",filename="Vis{}".format(str(i)),electrodesToPlot= electrodeIndices,electrodesUsedForTraining=electrodeIndices,nrOfLevels=i)

#signal_plots.plotSignalLR(inputSignals=trainX,groundTruth=trainY,prediction=prediction,maxValue=100,plotSignalsSeperatelyBool=True,meanBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesUsedForTraining=electrodeIndices,electrodesToPlot=electrodeIndices)
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],meanBool=True,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],filename="SignalVis_NoMean",meanBool=False,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")