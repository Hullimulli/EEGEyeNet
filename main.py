import sys
import time
import logging
from config import config, create_folder
from utils.tables_utils import print_table
from Joels_Files.AnalEyeZor import AnalEyeZor
import pandas as pd
import numpy as np
from utils import IOHelper
from tqdm import tqdm

"""
Main entry of the program
Creates the logging files, loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""

def main():
    # Setting up logging

    #asdf = AnalEyeZor(task='LR_task',dataset='antisaccade',preprocessing='max', trainBool=False, path="/Users/Hullimulli/Documents/ETH/SA2/run1/",models=["InceptionTime"],featureExtraction=True)
    local = False


    #asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='min', trainBool=False, models=["InceptionTime"],featureExtraction=False)
    if local:

        def plot(filename,modelNames,colour,PFIIndexNames=None):
            asdf = AnalEyeZor(task='Position_task', dataset='dots', preprocessing='min', trainBool=False,
                             path="Position_All/", models=modelNames, featureExtraction=False)

            if PFIIndexNames is None:
                for modelName in modelNames:
                    lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_'+filename+'.csv', usecols=[modelName]).to_numpy()
                    asdf.electrodeBarPlot(values=lossValues, colour='orange',name="Electrode_Loss_"+filename)
                    asdf.topoPlot(lossValues,cmap=colour,filename="Topoplot_"+filename,pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/",epsilon=0.01)
                    asdf.electrodePlot(colourValues=asdf.colourCode(values=np.squeeze(lossValues),colourMap=colour,epsilon=0.01),filename='Electrode_Losses_'+filename,alpha=1, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            else:
                for name in PFIIndexNames:
                    lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_' + filename + '.csv',
                                             usecols=[name]).to_numpy()
                    asdf.electrodeBarPlot(values=lossValues, colour='orange', name="Electrode_Loss_" + filename)
                    asdf.topoPlot(lossValues, cmap=colour, filename="Topoplot_" + filename,
                                  pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/", epsilon=0.01)
                    asdf.electrodePlot(
                        colourValues=asdf.colourCode(values=np.squeeze(lossValues), colourMap=colour, epsilon=0.01),
                        filename='Electrode_Losses_' + filename, alpha=1,
                        pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        #plot("InceptionTime_angle","InceptionTime","Oranges")
        #plot("InceptionTime_amplitude", "InceptionTime", "Oranges")

        def transformData(path,model,task):
            dataset='dots'
            electrodes = np.arange(129) + 1
            if "2" in path.lower():
                electrodes = np.array([1, 32])
            if task == 'LR':
                dataset='antisaccade'
            asdf = AnalEyeZor(task=task+'_task', electrodes=electrodes, dataset=dataset, preprocessing='min', trainBool=False,
                             path=path, models=[model], featureExtraction=False)
            #asdf.pca()
            #asdf.activationMaximization(model,epochs=1, steps=5000,componentAnalysis="PCA",dimensions=1, referenceIndices=np.asarray([23692,23693]), referenceElectrodes=np.asarray([32]),initTensor="Avg", filenamePostfix="_Lin", derivativeWeight=100000)
            #asdf.customSignal("Step")
            #asdf.customSignal("ContStepConfused",amplitude=0)
            asdf.attentionVisualization(model,filename="AttentionVis_ActMaxGen",componentAnalysis="PCA",method="Saliency",dimensions=1,run=1,dataIndices=np.asarray([23692,23693]),dataType="activationMaximization")
            asdf.plotSignal('InceptionTime', np.array([1, 32]),filename="AttentionVis_Signal_ActMaxGen",maxValue=100,dataType="activationMaximization")
            fdsa = 0


        transformData("LRMin_InceptionTime_Top2/", "InceptionTime", 'LR')
        #transformData("LRMin_InceptionTime_All/", "InceptionTime", 'LR')
        #transformData("Position_All/", "PyramidalCNN", 'Position')
        def showDataSimple(path,model,name,task,electrodes=np.arange(129)+1,run=1,colourMap='gist_rainbow',nrOfPoints=10,tresh=0.0,maxValue=100):
            dataset='dots'
            electrodesNetwork = np.arange(129)+1
            if "2" in path.lower():
                electrodesNetwork = np.array([1, 32])
            if task == 'LR':
                dataset='antisaccade'
            asdf = AnalEyeZor(task=task+'_task', dataset=dataset, preprocessing='min', trainBool=False,
                             path=path, models=[model], featureExtraction=False, electrodes=electrodesNetwork)
            asdf.plotSignal(model, electrodes, colourMap=colourMap, run=run, nrOfPoints=nrOfPoints,
                            filename=name, plotTresh=tresh, maxValue=maxValue, nrOfLevels=4, percentageThresh=5,
                            meanBool=False, componentAnalysis="PCA", splitAngAmpBool=True, dimensions=1)

        def countIDs():
            #indices = np.load("./Joels_Files/WeirdsZeros.npy")
            asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='min', trainBool=False,
                             path="LRMin_InceptionTime_Top2/", models=['InceptionTime'], featureExtraction=False, electrodes=np.array([1, 32]))
            indices = asdf.findDataPoints(componentAnalysis="", dimensions=1)
            #asdf.plotSignal('InceptionTime', np.array([1, 32]), colourMap='gist_rainbow', run=1, nrOfPoints=20000,
            #                filename="OneSignal", plotTresh=0,
            #                maxValue=100, nrOfLevels=2, percentageThresh=5,
            #                meanBool=False, componentAnalysis="", splitAngAmpBool=True, dimensions=1,
            #                activationMaximizationBool=False, plotSignalsSeperatelyBool=True, plotMovementBool=True,scaleModification=1,specificDataIndices=indices)
            asdf.plotSignal('InceptionTime', np.array([1, 32]), colourMap='gist_rainbow', run=1, nrOfPoints=20000,
                            filename="OneSignal", plotTresh=0,
                            maxValue=100, nrOfLevels=2, percentageThresh=5,
                            meanBool=False, componentAnalysis="", splitAngAmpBool=True, dimensions=1,
                            dataType="", plotSignalsSeperatelyBool=True, plotMovementBool=False,scaleModification=1,specificDataIndices=indices)
            asdf.plotSignal('InceptionTime', np.array([1, 32]), colourMap='gist_rainbow', run=1, nrOfPoints=20000,
                            filename="OneSignal_PCA", plotTresh=0,
                            maxValue=100, nrOfLevels=2, percentageThresh=5,
                            meanBool=False, componentAnalysis="PCA", splitAngAmpBool=True, dimensions=1,
                            dataType="", plotSignalsSeperatelyBool=True, plotMovementBool=False,scaleModification=1,specificDataIndices=indices)

        #countIDs()
        def showData(path,model,name,task,electrodes=np.arange(129)+1,run=1,colourMap='gist_rainbow',nrOfPoints=10,tresh=0.0,maxValue=100):
            dataset='dots'
            electrodesNetwork = np.arange(129)+1
            if "2" in path.lower():
                electrodesNetwork = np.array([1, 32])
            if task == 'LR':
                dataset='antisaccade'
            asdf = AnalEyeZor(task=task+'_task', dataset=dataset, preprocessing='min', trainBool=False,
                             path=path, models=[model], featureExtraction=False, electrodes=electrodesNetwork)
            asdf.plotSignal(model, electrodes, colourMap=colourMap, run=run, nrOfPoints=20000,
                            filename=name + "ActivationMax", plotTresh=tresh,
                            maxValue=maxValue, nrOfLevels=2, percentageThresh=5,
                            meanBool=False, componentAnalysis="", splitAngAmpBool=True, dimensions=1,
                            dataType="activationMaximization", scaleModification=1,postfix="_Lin")
            #asdf.plotSignal(model, electrodes, colourMap=colourMap, run=run, nrOfPoints=nrOfPoints, filename=name+"_PCA10",plotTresh=tresh, maxValue=maxValue, nrOfLevels=16, meanBool=True, componentAnalysis="PCA",splitAngAmpBool=True, dimensions=10, activationMaximizationBool=False)
            #asdf.plotSignal(model, electrodes, colourMap=colourMap, run=run, nrOfPoints=nrOfPoints, filename=name+"_PCA1",plotTresh=tresh, maxValue=maxValue, nrOfLevels=16, meanBool=True, componentAnalysis="PCA",splitAngAmpBool=True, dimensions=1, activationMaximizationBool=False)
            #asdf.movie(1,maxValue=25,cmap='seismic')

        #showData("LRMin_InceptionTime_All/", "InceptionTime", "SignalVisualisation", 'LR',electrodes=np.arange(129) + 1, nrOfPoints=20000, tresh=0, maxValue=70)
        #showData("LRMin_InceptionTime_Top2/","InceptionTime","SignalVisualisation",'LR',electrodes=np.array([1,32]),nrOfPoints=20000,tresh=0,maxValue=70)
        #showData("Position_All/", "PyramidalCNN", "SignalVisualisation", 'Position',electrodes=np.arange(129)+1, nrOfPoints=20000, tresh=0, maxValue=70)
        #showDataSimple("Direction_All/", "PyramidalCNN", "SignalVisualisation", 'Direction', electrodes=np.array([1,32]),nrOfPoints=20000, tresh=0, maxValue=70)
        def customPlot():
            asdf = AnalEyeZor(task='Direction_task', dataset='dots', preprocessing='min', trainBool=False,
                             path="Direction_All/", models=["Xception"], featureExtraction=False)
            goodElectrodeIndices = np.zeros(129) + 45
            array = np.array([1, 32])
            goodElectrodeIndices[array - 1] = 25
            array = np.array([125, 128])
            goodElectrodeIndices[array - 1] = 30
            array = np.array([17, 38, 121])
            goodElectrodeIndices[array - 1] = 35
            array = np.array([2, 3, 8, 9, 14, 21, 22, 23, 25, 26, 27, 33, 43, 120, 122, 123])
            goodElectrodeIndices[array-1] = 40
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap='nipy_spectral'),
                               filename='Configuration', alpha=0.4,
                               pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        #customPlot()
        def table(filename,directory,caption,scale):
            asdf = AnalEyeZor(task='Direction_task', dataset='dots', preprocessing='min', trainBool=False,
                             path=directory, models=["InceptionTime"], featureExtraction=False)
            asdf.generateTable(modelFileName=filename,addNrOfParams=False,filename=filename[:-4],caption=caption,scale=scale)

        def train(filename,electrodes,prep):
            asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing=prep, trainBool=True
                              , models=["InceptionTime"], electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)

        #table("statistics_all.csv","Position_All/",caption="Performance of each Network with all Configuration for the Position Task (RMSE).",scale=0.5)
        #table("statistics_angle_all.csv", "Position_All/", caption="Performance of each Network with all Configuration for the Angular Part of the Direction Task (RMSE).",scale=180 / float(np.pi))
        #table("statistics_angle_all.csv", "Position_All/",caption="Performance of each Network with all Configuration for the Angular Part of the Direction Task (RMSE).",scale=1)
        #table("statistics_amplitude_all.csv", "Position_All/", caption="Performance of each Network with all Configuration for the Amplitude Part of the Direction Task (RMSE).",scale=0.5)

        def visualizeTraining(task, name,network,columns):
            asdf = AnalEyeZor(task=task+'_task', dataset='dots', preprocessing='min', trainBool=False,
                              path=task+"_"+network+"/", models=["Xception","CNN","PyramidalCNN","InceptionTime","EEGNet"], featureExtraction=False)
            asdf.plotTraining(modelFileName=name, filename="Training_"+name[:-4],columns=columns)

        def combineResults(directories):
            asdf = AnalEyeZor(task='Position_task', dataset='dots', preprocessing='min', trainBool=False,
                              path="Position_All/", models=["Xception"], featureExtraction=False)
            asdf.combineResults("statistics.csv",directories,filename="Position_Xception_Statistics",nameStartIndex=9,addNrOfParams=True)
            asdf.generateTable("Position_Xception_Statistics.csv",filename='Position_Xception_Statistics',addNrOfParams=False)

        #directories = ["Direction_PyramidalCNN_SideFronts/","Direction_PyramidalCNN_Top2_Amplitude/","Direction_PyramidalCNN_Top6_Amplitude/","Direction_PyramidalCNN_Top2_Angle/","Direction_PyramidalCNN_Top6_Angle/"]
        #directories = ["Position_Xception_SideFronts/", "Position_Xception_Top2/", "Position_Xception_Top6/"]
        #combineResults(directories)

        def visualize(task, type, electrodes, modelNames, colour):
            asdf = AnalEyeZor(task=task+'_task', dataset='dots', preprocessing='min', trainBool=False,
                              path=task+"_"+type+"/", models=modelNames, electrodes=electrodes,featureExtraction=False)
            asdf.visualizePrediction(modelNames=modelNames, nrOfruns=5, filename="Visualisation_"+type)

            goodElectrodeIndices = np.zeros(129)
            goodElectrodeIndices[electrodes-1] = 1
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap=colour),
                               filename='Configuration', alpha=0.4,
                               pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            if task == "Position":
                for modelName in modelNames:
                    for i in range(1,6):
                        asdf.plotTraining(modelFileName=modelName + "_" + str(i) + ".csv", filename="Training_" + modelName + "_run" + str(i), columns=["Loss", "Val_Loss"])
            else:
                for modelName in modelNames:
                    for i in range(1,6):
                        asdf.plotTraining(modelFileName=modelName+"_angle"+str(i)+".csv", filename="Training_"+modelName+"_angle"+str(i), columns=["Loss","Accuracy"])
                        asdf.plotTraining(modelFileName=modelName + "_amplitude" + str(i) + ".csv",filename="Training_" + modelName + "_amplitude" + str(i), columns=["Loss","Val_Loss"])

        #visualize("Position_PyramidalCNN","PyramidalCNN")
        #visualize("Direction", "Xception")
        #names = ["InceptionTime_SideFronts","InceptionTime_Top2_Amplitude","InceptionTime_Top6_Amplitude","InceptionTime_Top2_Angle","InceptionTime_Top6_Angle"]
        #electrodes = [np.array([  1,  2,  3,  8,  9, 14, 21, 22, 23, 25, 26, 27, 32, 33, 38, 43,120,121, 122,123,125,128]),np.array([ 27,123]),np.array([ 27, 38, 43,120,121,123]),np.array([125,128]),np.array([  1, 32, 38,121,125,128])]

        top2Amp = np.array([1,32])
        top2Ang = np.array([125,128])
        top4 = np.array([1,32,125,128])
        top7 = np.array([1,17,32,38,121,125,128])
        sideFronts = np.array([1,2,3,8,9,14,17,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128])

        #plot("Average_Amp",["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"],colour='Reds',PFIIndexNames=["Average"])
        #plot("Average_Ang", ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], colour='Reds',PFIIndexNames=["Average"])
        #visualize("Direction", "All", np.arange(129)+1, ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Reds")
        #visualize("Direction", "Top2Amp", top2Amp,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Reds")
        #visualize("Direction", "Top2Ang", top2Ang,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Greens")
        #visualize("Direction", "Top4", top4,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Purples")
        #visualize("Direction", "Top7", top7,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Oranges")
        #visualize("Direction", "SideFronts", sideFronts,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Blues")

        #plot("Average", ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], colour='Reds',PFIIndexNames=["Average"])
        #visualize("Position", "All", np.arange(129)+1, ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Reds")
        #visualize("Position", "Top2", top2Ang,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Greens")
        #visualize("Position", "Top4", top4,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Purples")
        #visualize("Position", "Top7", top7,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Oranges")
        #visualize("Position", "SideFronts", sideFronts,["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], "Blues")

        #names = ["InceptionTime_SideFronts", "InceptionTime_Top2", "InceptionTime_Top6"]
        #electrodes = [np.array([1, 2, 3, 8, 9, 14, 21, 22, 23, 25, 26, 27, 32, 33, 38, 43, 120, 121, 122, 123, 125, 128]),np.array([ 1,32]),np.array([  1, 38, 32,121,125,128])]





    if not local:

        def train(filename,network,electrodes,prep,task,trainBool=True):
            dataset = "dots"
            if task == "LR_task":
                dataset="antisaccade"
            asdf = AnalEyeZor(task=task, dataset=dataset, preprocessing=prep, trainBool=trainBool
                               ,models=network, electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
            del asdf


        def PFI(filename,network,electrodes,prep,task,trail=False, trainBool=True):
            dataset = "dots"
            if task == "LR_task":
                dataset="antisaccade"
            asdf = AnalEyeZor(task=task, dataset=dataset, preprocessing=prep, trainBool=trainBool
                               ,path=filename,models=network, electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
            if trail:
                asdf.PFI(saveTrail="_amplitude",nameSuffix="_"+network[0]+"_amplitude")
                asdf.PFI(saveTrail="_angle",nameSuffix="_"+network[0]+"_angle")
            else:
                asdf.PFI(nameSuffix="_"+network[0])
            del asdf

        #train("LR_PyramidalCNN",["PyramidalCNN"], 1 + np.arange(129), 'min', "LR_task")
        train("LR_CNN_Top2",["CNN"], np.array([1,32]), 'min', "LR_task")
        #train('Direction_Xception_Top2_Amplitude', ["Xception"], np.array([27,123]), 'min', "Direction_task")
        #train('Direction_Xception_Top2_Angle', ["Xception"], np.array([1,32]), 'min', "Direction_task")
        #train('Direction_Xception_Top4_Angle', ["Xception"], np.array([1,32,125,128]), 'min', "Direction_task")
        #train('Direction_Xception_Top4_Amplitude', ["Xception"], np.array([27,38,121,123]), 'min', "Direction_task")
        #train('Direction_Xception_SideFronts', ["Xception"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")
        #train('Position_Xception_Top2', ["Xception"], np.array([1,32]), 'min', "Position_task")
        #train('Position_Xception_Top6', ["Xception"], np.array([1,32,38,121,125,128]), 'min', "Position_task")
        #train('Position_Xception_SideFronts', ["Xception"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Position_task")

        #train('Position_CNN_Top2', ["CNN"], np.array([1,32]), 'min', "Position_task")
        #train('Position_CNN_Top6', ["CNN"], np.array([1,32,38,121,125,128]), 'min', "Position_task")
        #train('Position_CNN_SideFronts', ["CNN"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Position_task")

        #train('Position_EEGNet_Top2', ["EEGNet"], np.array([1,32]), 'min', "Position_task")
        #train('Position_EEGNet_Top6', ["EEGNet"], np.array([1,14,21,32,125,128]), 'min', "Position_task")
        #train('Position_EEGNet_SideFronts', ["EEGNet"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Position_task")

        #train('Direction_EEGNet_Top2_Amplitude', ["EEGNet"], np.array([38,121]), 'min', "Direction_task")
        #train('Direction_EEGNet_Top2_Angle', ["EEGNet"], np.array([1,32]), 'min', "Direction_task")
        #train('Direction_EEGNet_Top6_Angle', ["EEGNet"], np.array([1,27,32,123,125,128]), 'min', "Direction_task")
        #train('Direction_EEGNet_Top6_Amplitude', ["EEGNet"], np.array([27,38,43,120,121,123]), 'min', "Direction_task")
        #train('Direction_EEGNet_SideFronts', ["EEGNet"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")

        #train('Direction_CNN_Top2_Amplitude', ["CNN"], np.array([27,123]), 'min', "Direction_task")
        #train('Direction_CNN_Top2_Angle', ["CNN"], np.array([125,128]), 'min', "Direction_task")
        #train('Direction_CNN_Top6_Angle', ["CNN"], np.array([1,2,26,32,125,128]), 'min', "Direction_task")
        #train('Direction_CNN_Top6_Amplitude', ["CNN"], np.array([3,23,27,43,120,123]), 'min', "Direction_task")
        #train('Direction_CNN_SideFronts', ["CNN"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")

        #train('Position_PyramidalCNN_Top2', ["PyramidalCNN"], np.array([1,32]), 'min', "Position_task")
        #train('Position_PyramidalCNN_Top6', ["PyramidalCNN"], np.array([1,14,21,32,125,128]), 'min', "Position_task")
        #train('Position_PyramidalCNN_SideFronts', ["PyramidalCNN"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Position_task")

        #train('Direction_PyramidalCNN_Top2_Amplitude', ["PyramidalCNN"], np.array([27,123]), 'min', "Direction_task")
        #train('Direction_PyramidalCNN_Top2_Angle', ["PyramidalCNN"], np.array([125,128]), 'min', "Direction_task")
        #train('Direction_PyramidalCNN_Top6_Angle', ["PyramidalCNN"], np.array([1,2,26,32,125,128]), 'min', "Direction_task")
        #train('Direction_PyramidalCNN_Top6_Amplitude', ["PyramidalCNN"], np.array([1,27,32,43,120,123]), 'min', "Direction_task")
        #train('Direction_PyramidalCNN_SideFronts', ["PyramidalCNN"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")

        #train('Position_InceptionTime_Top2', ["InceptionTime"], np.array([1,32]), 'min', "Position_task")
        #train('Position_InceptionTime_Top6', ["InceptionTime"], np.array([1,38,32,121,125,128]), 'min', "Position_task")
        #train('Position_InceptionTime_SideFronts', ["InceptionTime"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Position_task")

        #train('Direction_InceptionTime_Top2_Amplitude', ["InceptionTime"], np.array([27,123]), 'min', "Direction_task")
        #train('Direction_InceptionTime_Top2_Angle', ["InceptionTime"], np.array([125,128]), 'min', "Direction_task")
        #train('Direction_InceptionTime_Top6_Angle', ["InceptionTime"], np.array([1,32,38,121,125,128]), 'min', "Direction_task")
        #train('Direction_InceptionTime_Top6_Amplitude', ["InceptionTime"], np.array([27,38,43,120,121,123]), 'min', "Direction_task")
        #train('Direction_InceptionTime_SideFronts', ["InceptionTime"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")


        #train('Position_All', ["Xception","CNN","PyramidalCNN","InceptionTime","EEGNet"], 1 + np.arange(129), 'min', "Position_task", trail=True,trainBool=True)
        #PFI('Direction_CNN_All/', ["CNN"], 1 + np.arange(129), 'min', "Direction_task", trail=True, trainBool=True)
        #PFI('Position_CNN_All/', ["CNN"], 1 + np.arange(129), 'min', "Position_task", trail=False, trainBool=True)
        #PFI('Direction_InceptionTime_All/', ["InceptionTime"], 1 + np.arange(129), 'min', "Direction_task", trail=True, trainBool=True)
        #PFI('Position_InceptionTime_All/', ["InceptionTime"], 1 + np.arange(129), 'min', "Position_task", trail=False, trainBool=True)
        #PFI('Direction_PyramidalCNN_All/', ["PyramidalCNN"], 1 + np.arange(129), 'min', "Direction_task", trail=True, trainBool=True)
        #PFI('Position_PyramidalCNN_All/', ["PyramidalCNN"], 1 + np.arange(129), 'min', "Position_task", trail=False, trainBool=True)
        #PFI('Direction_EEGNet_All/', ["EEGNet"], 1 + np.arange(129), 'min', "Direction_task", trail=True, trainBool=True)
        #PFI('Position_EEGNet_All/', ["EEGNet"], 1 + np.arange(129), 'min', "Position_task", trail=False, trainBool=True)
        #PFI('Direction_Xception_All/', ["Xception"], 1 + np.arange(129), 'min', "Direction_task", trail=True, trainBool=True)
        #PFI('Position_Xception_All/', ["Xception"], 1 + np.arange(129), 'min', "Position_task", trail=False, trainBool=True)

        #Direction
        top2Amp = np.array([1,32])
        top2Ang = np.array([125,128])
        top4 = np.array([1,32,125,128])
        top7 = np.array([1,17,32,38,121,125,128])
        sideFronts = np.array([1,2,3,8,9,14,17,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128])

        #Position
        top2 = np.array([125,128])
        top4 = np.array([1,32,125,128])
        top7 = np.array([1,17,32,38,121,125,128])
        sideFronts = np.array([1,2,3,8,9,14,17,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128])

        #train('Direction_Top2Amp', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top2Amp, 'min',"Direction_task")
        #train('Direction_Top2Ang', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top2Ang, 'min', "Direction_task")
        #train('Direction_Top4', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top4, 'min', "Direction_task")
        #train('Direction_Top7', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top7, 'min', "Direction_task")
        #train('Direction_SideFronts', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], sideFronts, 'min', "Direction_task")

        #train('Position_Top2', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top2, 'min',"Position_task")
        #train('Position_Top4', ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], top4, 'min',"Position_task")
        #train('Position_Top7', ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], top7, 'min',"Position_task")
        #train('Position_SideFronts', ["InceptionTime", "EEGNet", "CNN", "PyramidalCNN", "Xception"], sideFronts, 'min',"Position_task")


        #for i in ["EEGNet"]:
        #    PFI('Direction_All/', [i], 1 + np.arange(129), 'min', "Direction_task", trail=True,
        #          trainBool=False)
        #    PFI('Position_All/', [i], 1 + np.arange(129), 'min', "Position_task", trail=False,
        #          trainBool=False)

    #asdf.plotTraining(name="InceptionTime1_Training", modelFileName="InceptionTime_1.csv",columns=["Loss", "Val_Loss"])

    #start_time = time.time()

    # For being able to see progress that some methods use verbose (for debugging purposes)
    #asdf.chooseModel(["InceptionTime"], trainBool=False, saveModelBool=False, path="inceptionTimeAntisaccade/")
    #asdf.chooseModel(["InceptionTime"], trainBool=False, saveModelBool=False)
    #asdf.PFI()
    #Load the data
    #trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)

    #Start benchmark
    #benchmark(trainX, trainY)
    #directory = 'results/standardML'
    #print_table(directory, preprocessing='max')

    #logging.info("--- Runtime: %s seconds ---" % (time.time() - start_time))
    #logging.info('Finished Logging')

if __name__=='__main__':
    main()
