import numpy as np

from benchmark import benchmark
from utils import IOHelper
from config import config
from benchmark import split
from Joels_Files.plotFunctions import electrode_plots
from Joels_Files.simpleRegressor import simpleDirectionRegressor
from Joels_Files.plotFunctions import signal_plots, prediction_visualisations
asdf = np.arange(129)
#electrode_plots.electrodeBarPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#colours = electrode_plots.colourCode(-asdf)
#electrode_plots.electrodePlot(colours,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")

#electrode_plots.topoPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
electrodeIndices = np.array([1,17,32])
#benchmark()

#amps = np.array([500,700,900])
#pred = np.array([[300,400,600],[400,300,700],[400,300,700],[400,300,700],[400,300,700]])
#pred = np.expand_dims(pred,axis=1)
#eeh = prediction_visualisations.visualizePredictionAmplitude(groundTruth=amps,directory = "./",prediction=pred,modelNames=["Doodoo","Poopoo","Gaga","Duudu","fjs"],saveBool=False)
#asdf = 0
#directory = "/Users/Hullimulli/Documents/ETH/SA2/debugFolder/"
#trainIndices, valIndices, testIndices = split(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:5000,0], 0.7, 0.15, 0.15)

simpleDirectionRegressor(electrodeIndices)

#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:,:, electrodeIndices-1]
#signal_math.pca(trainX,filename="test",directory=directory)
#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:,:, electrodeIndices-1]
#trainX=trainX-trainX[:,:50].mean(axis=1, keepdims=True)
#eye = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:5000,:, 128]
#trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:,1]

#rightOnlyIndices = np.argwhere(np.squeeze(np.argwhere(np.absolute(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:,2]) <= 1*np.pi/4)))

#signal_plots.plotSignalAmplitude(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=electrodeIndices,electrodesUsedForTraining=electrodeIndices,filename="SignalVisualitation_Amplitude",nrOfLevels=8)

#signal_plots.plotSignalAngle(inputSignals=signal_math.pcaDimReduction(trainX,file=directory+"test.npy"),groundTruth=trainY,directory=directory,filename="Vis",electrodesToPlot= electrodeIndices,electrodesUsedForTraining=electrodeIndices,nrOfLevels=8)

#signal_plots.plotSignalLR(inputSignals=trainX,groundTruth=trainY,prediction=None,filename="SigVis_Detrend20",maxValue=10000,plotSignalsSeperatelyBool=False,meanBool=True,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesUsedForTraining=electrodeIndices,electrodesToPlot=electrodeIndices)
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],meanBool=True,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],filename="SignalVis_NoMean",meanBool=False,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")