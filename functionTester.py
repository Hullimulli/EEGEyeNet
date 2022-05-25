import numpy as np
import tensorflow.keras as keras
from benchmark import benchmark
from utils import IOHelper
from config import config
import os
from benchmark import split
from Joels_Files.plotFunctions import electrode_plots
from Joels_Files.simpleRegressor import simpleDirectionRegressor
from Joels_Files.plotFunctions import signal_plots, prediction_visualisations, attention_visualisations
from Joels_Files.mathFunctions import electrode_math
from Joels_Files.helperFunctions import predictor
import pandas as pd

def getTestIndices():
    from benchmark import split
    ids = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:, 0]
    train, val, test = split(ids, 0.7, 0.15, 0.15)
    return test


asdf = np.arange(129)
#electrode_plots.electrodeBarPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#colours = electrode_plots.colourCode(-asdf)
#electrode_plots.electrodePlot(colours,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")

#electrode_plots.topoPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
directory = "/Users/Hullimulli/Documents/ETH/SA2/debugFolder/"
architectures = ["InceptionTime","EEGNet","PyramidalCNN","CNN","Xception"]
#indices = np.squeeze(getTestIndices())
#predictions = np.load(os.path.join(directory,"Direction_task_Angle_All.npy"))
#trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices,2]

#prediction_visualisations.visualizePredictionAngle(directory=directory,groundTruth=trainY[:7],prediction=predictions[:,:,:7],modelNames=architectures,filename="Ang_All_PredVis")






tasks = ["DirectionTaskTop3","DirectionTaskTop2","DirectionTaskAll"]
electrodeIndices = [np.array([17,125,128]),np.array([125,128]),np.arange(129)+1]
for j,i in enumerate(tasks):
    experimentPath = "/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/"+i
    angleArchitectureBool=True
    filename = "Ang_"+i[13:]
    indices = np.squeeze(getTestIndices())
    # trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices,1]
    trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][indices, :, :]
    trainX = trainX[:, :, electrodeIndices[j] - 1]

    predictions = predictor.savePredictions(filename=filename, savePath=directory, inputSignals=trainX,
                              experimentFolderPath=experimentPath, architectures=architectures,
                              angleArchitectureBool=angleArchitectureBool)
    del trainX
    trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices, 2]
    prediction_visualisations.visualizePredictionAngle(directory=directory, groundTruth=trainY[:7],
                                                       prediction=predictions[:, :, :7], modelNames=architectures,
                                                       filename="PredVis_"+filename)


#benchmark()


#amps = np.array([500,700,900])
#pred = np.array([[300,400,600],[400,300,700],[400,300,700],[400,300,700],[400,300,700]])
#pred = np.expand_dims(pred,axis=1)
#eeh = prediction_visualisations.visualizePredictionAmplitude(groundTruth=amps,directory = "./",prediction=pred,modelNames=["Doodoo","Poopoo","Gaga","Duudu","fjs"],saveBool=False)
#asdf = 0

#trainIndices, valIndices, testIndices = split(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:5000,0], 0.7, 0.15, 0.15)
#rightOnlyIndices = np.squeeze(np.argwhere(np.absolute(IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:,2]) <= 1*np.pi/4))
#simpleDirectionRegressor(electrodeIndices)
#benchmark()
#pathlist = electrode_math.modelPathsFromBenchmark("/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/1651674753_Position_task_dots_min",["PyramidalCNN","Xception","InceptionTime","CNN"],angleArchitectureBool=False)
#indices = np.squeeze(getTestIndices())
#trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices,1]
#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][indices,:,:]
#trainX = trainX[:,:,electrodeIndices-1]


#predictor.savePredictions(filename=filename,savePath=directory,inputSignals=trainX,experimentFolderPath=experimentPath,architectures=architectures,angleArchitectureBool=angleArchitectureBool)

#losses = electrode_math.PFI(inputSignals=trainX,groundTruth=trainY,loss='angle-loss', directory=directory,modelPaths=[pathlist[0]],iterations=1,filename='PFI_Original')
#base = electrode_math.gradientPFI(inputSignals=trainX,groundTruth=trainY,loss='mse', directory=directory,modelPaths=pathlist)
#electrode_plots.topoPlot(base,directory=directory,filename="SaliencyPFI_Pos",cmap='Reds')
#model = keras.models.load_model(pathlist[0], compile=False)
#grads = attention_visualisations.saliencyMap(model=model,loss='mse',inputSignals=trainX,groundTruth=trainY)
#attention_visualisations.plotSaliencyMap(inputSignals=trainX,groundTruth=trainY,gradients=grads,directory=directory,electrodesToPlot=np.array([1,32]))
#signal_math.pca(trainX,filename="test",directory=directory)
#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:,:, electrodeIndices-1]
#trainX=trainX-trainX[:,:50].mean(axis=1, keepdims=True)
#eye = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:5000,:, 128]



#signal_plots.plotSignalAmplitude(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=electrodeIndices,electrodesUsedForTraining=electrodeIndices,filename="ForSunday_SignalVisualitation_Amplitude",nrOfLevels=6,maxValue=200)
#signal_plots.plotSignalAngle(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=electrodeIndices,electrodesUsedForTraining=electrodeIndices,filename="ForSunday_SignalVisualitation_Angle",nrOfLevels=8,maxValue=100)
#signal_plots.plotSignalPosition(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=electrodeIndices,electrodesUsedForTraining=electrodeIndices,filename="Back_SignalVisualitation_Pos",nrOfLevels=9,maxValue=100)
#signal_plots.plotSignalAngle(inputSignals=signal_math.pcaDimReduction(trainX,file=directory+"test.npy"),groundTruth=trainY,directory=directory,filename="Vis",electrodesToPlot= electrodeIndices,electrodesUsedForTraining=electrodeIndices,nrOfLevels=8)

#signal_plots.plotSignalLR(inputSignals=trainX,groundTruth=trainY,prediction=None,filename="SigVis_Detrend20",maxValue=10000,plotSignalsSeperatelyBool=False,meanBool=True,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesUsedForTraining=electrodeIndices,electrodesToPlot=electrodeIndices)
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],meanBool=True,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],filename="SignalVis_NoMean",meanBool=False,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")