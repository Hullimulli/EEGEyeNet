import numpy as np

from utils import IOHelper
from config import config
from benchmark import split
from Joels_Files.plotFunctions import electrode_plots
from Joels_Files.plotFunctions import signal_plots, signal_math, prediction_visualisations
asdf = np.arange(129)
#electrode_plots.electrodeBarPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#colours = electrode_plots.colourCode(-asdf)
#electrode_plots.electrodePlot(colours,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")

#electrode_plots.topoPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
electrodeIndices = np.array([32])
directory = "/Users/Hullimulli/Documents/ETH/SA2/debugFolder/"
#trainIndices, valIndices, testIndices = split(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:5000,0], 0.7, 0.15, 0.15)
#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:300,:, electrodeIndices-1]
#signal_math.pca(trainX,filename="test",directory=directory)
#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][300:500,:, electrodeIndices-1]
#eye = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:5000,:, 128]
#trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][300:500,2]
#signal_plots.plotSignalAngle(inputSignals=signal_math.pcaDimReduction(trainX,file=directory+"test.npy"),groundTruth=trainY,directory=directory,filename="Vis",electrodesToPlot= electrodeIndices,electrodesUsedForTraining=electrodeIndices,nrOfLevels=8)
groundTruth = np.array([400,200,600])
prediction = np.array([[[250,180,580],[255,220,660]]])
modelNames = ["StupidModel"]
prediction_visualisations.visualizePredictionAmplitude(groundTruth=groundTruth,prediction=prediction,modelNames=modelNames,directory=directory)

#signal_plots.plotSignalLR(inputSignals=trainX,groundTruth=trainY,prediction=prediction,maxValue=100,plotSignalsSeperatelyBool=True,meanBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesUsedForTraining=electrodeIndices,electrodesToPlot=electrodeIndices)
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],meanBool=True,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],filename="SignalVis_NoMean",meanBool=False,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")