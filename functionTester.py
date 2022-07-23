import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from benchmark import benchmark
from utils import IOHelper
from config import config
import os
from benchmark import split
from Joels_Files.plotFunctions import electrode_plots
from Joels_Files.simpleRegressor import simpleDirectionRegressor
from Joels_Files.plotFunctions import signal_plots, prediction_visualisations, attention_visualisations
from Joels_Files.mathFunctions import electrode_math
from Joels_Files.helperFunctions import predictor, latex, modelLoader
import pandas as pd
from tqdm import tqdm
from copy import copy
from tempMultiCNN import getModel

def getTestIndices():
    from benchmark import split
    ids = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:, 0]
    train, val, test = split(ids, 0.7, 0.15, 0.15)
    return test

def getValIndices():
    from benchmark import split
    ids = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][:, 0]
    train, val, test = split(ids, 0.7, 0.15, 0.15)
    return val


asdf = np.arange(129)
directory = "/Users/Hullimulli/Documents/ETH/SA2/debugFolder/"

#electrode_plots.electrodeBarPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#colours = electrode_plots.colourCode(-asdf)
#electrode_plots.electrodePlot(colours,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")

#electrode_plots.topoPlot(asdf,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")

#architectures = ["InceptionTime","EEGNet","PyramidalCNN","CNN","Xception"]
#indices = np.squeeze(getTestIndices())
#predictions = np.load(os.path.join(directory,"Direction_task_Angle_All.npy"))
#trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices,2]

#prediction_visualisations.visualizePredictionAngle(directory=directory,groundTruth=trainY[:7],prediction=predictions[:,:,:7],modelNames=architectures,filename="Ang_All_PredVis")






# tasks = ["DirectionTaskTop3Old","DirectionTaskTop2Old","DirectionTaskTop3","DirectionTaskTop2"]
# electrodeIndices = [np.array([1,17,32]),np.array([1,32]),np.array([17,125,128]),np.array([125,128])]
# for j,i in enumerate(tasks):
#     experimentPath = "/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/"+i
#     angleArchitectureBool=True
#     filename = "Pos_"+i[12:]
#     indices = np.squeeze(getTestIndices())
#     #temp,predictionsSVM = simpleDirectionRegressor(electrodes=[electrodeIndices[j]])
#     #asdf = np.tile(predictionsSVM,(1,5,1))
#     # trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices,1]
#     trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][indices, :, :]
#     trainX = trainX[:, :, electrodeIndices[j] - 1]
#
#     # predictions = predictor.savePredictions(filename=filename, savePath=directory, inputSignals=trainX,
#     #                           experimentFolderPath=experimentPath, architectures=architectures,
#     #                           angleArchitectureBool=angleArchitectureBool)
#     #
#     trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][indices, 2]
#     # #predictions = np.concatenate((predictions,np.tile(predictionsSVM,(1,5,1))))
#     # prediction_visualisations.visualizePredictionPosition(directory=directory, groundTruth=trainY[:5],
#     #                                                    prediction=np.mean(predictions[:, :, :5],axis=1,keepdims=True), modelNames=architectures,
#     #                                                    filename="PredVis_"+filename)
#
#     pathlist = electrode_math.modelPathsFromBenchmark(
#         "/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/{}".format(i),
#         ["PyramidalCNN", "Xception", "InceptionTime", "CNN"], angleArchitectureBool=angleArchitectureBool)
#     for path in pathlist:
#         model = keras.models.load_model(path, compile=False)
#         grads = attention_visualisations.saliencyMap(model=model, loss='angle-loss', inputSignals=trainX[[0]],
#                                                      groundTruth=trainY[[0]])
#         attention_visualisations.plotSaliencyMap(inputSignals=trainX[[0]], groundTruth=trainY[[0]], gradients=grads,
#                                                 directory=directory, filename="AttentionVisualisation_{}_{}".format(i,os.path.basename(path)),
#                                                 electrodesToPlot=electrodeIndices[j], electrodesUsedForTraining=electrodeIndices[j])


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
#trainX = trainX[:,:,electrodeIndices-1]


#predictor.savePredictions(filename=filename,savePath=directory,inputSignals=trainX,experimentFolderPath=experimentPath,architectures=architectures,angleArchitectureBool=angleArchitectureBool)

#losses = electrode_math.PFI(inputSignals=trainX,groundTruth=trainY,loss='angle-loss', directory=directory,modelPaths=[pathlist[0]],iterations=1,filename='PFI_Original')
#base = electrode_math.gradientPFI(inputSignals=trainX,groundTruth=trainY,loss='angle-loss', directory=directory, modelPaths=pathlist, filename="PFI_Ang_Sal")
#electrode_plots.topoPlot(base,directory=directory,filename="SaliencyPFI_Ang",cmap='Purples',valueType = "Avg. Gradient")
pathlist = electrode_math.modelPathsFromBenchmark("/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/DirectionTaskAll",["PyramidalCNN","Xception","InceptionTime","CNN"],angleArchitectureBool=True)
#path = ['/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/torchModels/checkpoint/run1/ConvLSTM_nb_0.pth']
#model = modelLoader.returnTorchModel('/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/torchModels/checkpoint/run1/ConvLSTM_nb_0.pth')

model = keras.models.load_model(pathlist[1], compile=False)
print(model.summary())
config['framework'] = 'tensorflow'

#model = getModel('/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/attentionModel/Angle/checkpoint/CNNMultiTask_nb_0_.pth',type='angle')

#indices = np.squeeze(np.argwhere(getValIndices()))
with np.load(config['data_dir'] + config['all_EEG_file']) as f:
    trainX = f[config['trainX_variable']][[300]]
    trainY = f[config['trainY_variable']][[300],2]
start = 0
end = 250
#trainX[0,499,124] = 0
#trainX[0,start:end,124] = np.interp(np.arange(end-start), [start,end-1], [trainX[0,start,124],trainX[0,end-1,124]]) + np.random.normal(scale=5,size=end-start)
#trainX[0,start:end,124] = trainX[0,start:end,124] + np.random.normal(scale=5,size=end-start)
#trainX[0,start:end,124] = -trainX[0,start:end,124]


#model = modelLoader.returnTorchModel('/Users/Hullimulli/Documents/ETH/SA2/EEGEyeNet/runs/torchModels/checkpoint/run1/ConvLSTM_nb_0.pth')
# grads = attention_visualisations.fullGrad(model,trainX,trainY,'angle-loss',biasOnlyBool=True)
# maxValue = np.percentile(grads,40)
# print(maxValue)
# temp = copy(trainX)
# temp[np.where(grads<=maxValue)] = 0
# prediction = model(temp,training=False)
# attention_visualisations.plotSaliencyMap(inputSignals=temp,groundTruth=trainY,gradients=grads,directory=directory,electrodesToPlot=np.array([125]),filename="00Debug_FullGradBias_{}".format(np.array2string(np.squeeze(prediction))))
trainX = np.transpose(trainX,axes=(0,2,1))
grads = attention_visualisations.saliencyMap(model,trainX,trainY,'angle-loss',includeInputBool=True,absoluteValueBool = False)
grads = grads[0].T
electrode_plots.movie(grads,directory=directory)
# maxValue = np.percentile(grads,40)
# print(maxValue)
# temp = copy(trainX)
# temp[np.where(grads<=maxValue)] = 0
# prediction = model(temp,training=False)
#attention_visualisations.plotSaliencyMap(inputSignals=temp,groundTruth=trainY,gradients=grads,directory=directory,electrodesToPlot=np.array([125]),filename="00Debug_FullGrad_{}".format(np.array2string(np.squeeze(prediction))))

# grads = attention_visualisations.saliencyMap(model,trainX,trainY,'mse')
#
# attention_visualisations.plotSaliencyMap(inputSignals=trainX,groundTruth=trainY,gradients=grads,directory=directory,electrodesToPlot=np.array([125]),filename="Debug_Saliency")
#
# grads = attention_visualisations.saliencyMap(model, trainX, trainY, 'mse',includeInputBool=True)
#
# attention_visualisations.plotSaliencyMap(inputSignals=trainX, groundTruth=trainY, gradients=grads,
#                                          directory=directory, electrodesToPlot=np.array([125]),
#                                          filename="Debug_SaliencyInp")



#base = electrode_math.gradientBasedFI(inputSignals=trainX,groundTruth=trainY,modelPaths=path,directory=directory,filename='PFI_Torch_Grad',loss='mse',method="FullgradNoBias")
#electrode_plots.topoPlot(base,directory=directory,filename="FullGradPFI_Torch_Amp",cmap='Blues',valueType = "Avg. FullGrad")
#base = electrode_math.aggregatedLayerGradientBasedFI(inputSignals=trainX,groundTruth=trainY,loss='angle-loss', directory=directory, modelPaths=pathlist, filename="PFI_Ang_Sal")
#electrode_plots.topoPlot(base,directory=directory,filename="SaliencyPFI_Ang",cmap='Purples',valueType = "Avg. Full Gradient")
#grads = attention_visualisations.fullGradTensorflow(model,trainX,trainY,'angle-loss')

#attention_visualisations.plotSaliencyMap(inputSignals=trainX,groundTruth=trainY,gradients=grads,directory=directory,electrodesToPlot=np.array([125,128]),filename="OldAttentionVisualisation")
#signal_math.pca(trainX,filename="test",directory=directory)
#trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:,:, electrodeIndices-1]
#trainX=trainX-trainX[:,:50].mean(axis=1, keepdims=True)
#eye = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:5000,:, 128]

# latex.generateTable("/Users/Hullimulli/Documents/ETH/SA2/ResultsUpToDate/Statistics/AngleTask.csv",directory=directory,
#                     filename="angleTaskLatex",transposed=False,scale=100, caption="Performance of each Network for the Angle Task.")


#signal_plots.plotSignalAmplitude(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=electrodeIndices,electrodesUsedForTraining=electrodeIndices,filename="ForSunday_SignalVisualitation_Amplitude",nrOfLevels=6,maxValue=200)
#signal_plots.plotSignalAngle(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=np.array([1,32]),electrodesUsedForTraining=np.arange(129)+1,filename="ForSunday_SignalVisualitation_Angle",nrOfLevels=8,maxValue=100)
#signal_plots.plotSignalPosition(inputSignals=trainX,groundTruth=trainY,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesToPlot=electrodeIndices,electrodesUsedForTraining=electrodeIndices,filename="Back_SignalVisualitation_Pos",nrOfLevels=9,maxValue=100)
#signal_plots.plotSignalAngle(inputSignals=signal_math.pcaDimReduction(trainX,file=directory+"test.npy"),groundTruth=trainY,directory=directory,filename="Vis",electrodesToPlot= electrodeIndices,electrodesUsedForTraining=electrodeIndices,nrOfLevels=8)

#signal_plots.plotSignalLR(inputSignals=trainX,groundTruth=trainY,prediction=None,filename="SigVis_Detrend20",maxValue=10000,plotSignalsSeperatelyBool=False,meanBool=True,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",electrodesUsedForTraining=electrodeIndices,electrodesToPlot=electrodeIndices)
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],meanBool=True,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")
#signal_plots.plotEyeLR(eyeMovement=eye[indicesOfInterest],groundTruth=trainY[indicesOfInterest],filename="SignalVis_NoMean",meanBool=False,plotSignalsSeperatelyBool=False,directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/")