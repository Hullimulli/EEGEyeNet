import sys
import time
import logging
from config import config, create_folder
from utils.tables_utils import print_table
from Joels_Files.AnalEyeZor import AnalEyeZor
import pandas as pd
import numpy as np

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

        def plot(filename,model,colour):
            asdf = AnalEyeZor(task='Direction_task', dataset='dots', preprocessing='min', trainBool=False,
                             path="Direction_All/", models=[model], featureExtraction=False)
            lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_'+filename+'.csv', usecols=[model]).to_numpy()
            goodElectrodeIndices = np.zeros(np.squeeze(lossValues).shape)
            goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:2]] = 1
            asdf.electrodeBarPlot(values=lossValues, colour='orange',name="Electrode_Loss_"+filename)
            asdf.topoPlot(lossValues,cmap=colour,filename="Topoplot_"+filename,pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/",epsilon=0.01)
            asdf.electrodePlot(colourValues=asdf.colourCode(values=np.squeeze(lossValues),colourMap=colour,epsilon=0.01),filename='Electrode_Losses_'+filename,alpha=1, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices,colourMap=colour), filename='best2Electrode_'+filename, alpha=0.4, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:4]] = 1
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap=colour),
                              filename='best4Electrode_'+filename, alpha=0.4, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        #plot("InceptionTime_angle","InceptionTime","Oranges")
        #plot("InceptionTime_amplitude", "InceptionTime", "Oranges")
        def customPlot(array):
            asdf = AnalEyeZor(task='Direction_task', dataset='dots', preprocessing='min', trainBool=False,
                             path="Direction_All/", models=["Xception"], featureExtraction=False)
            goodElectrodeIndices = np.zeros(129)
            goodElectrodeIndices[array-1] = 1
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap='Reds'),
                               filename='Configuration', alpha=0.4,
                               pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        #customPlot()
        def table(filename,directory,model):
            asdf = AnalEyeZor(task='Position_task', dataset='dots', preprocessing='min', trainBool=False,
                             path=directory, models=model, featureExtraction=False)
            asdf.generateTable(modelFileName=filename,addNrOfParams=True,filename=filename[:-4])

        def train(filename,electrodes,prep):
            asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing=prep, trainBool=True
                              , models=["InceptionTime"], electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)

        #table("statistics.csv","Position_All/",["Xception","CNN","PyramidalCNN","InceptionTime","EEGNet"])
        #table("statistics_angle.csv", "Direction_All/", ["Xception","CNN","PyramidalCNN","InceptionTime","EEGNet"])

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

        asdf = AnalEyeZor(task='Direction_task', dataset='dots', preprocessing='min', trainBool=False,
                          path="Direction_All/", models=["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"],
                          featureExtraction=False)
        asdf.visualizePrediction(modelNames=["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], nrOfruns=5, filename="Visualisation_All_run")

        def visualize(task, model,run,electrodes, modelName,colour):
            asdf = AnalEyeZor(task=task+'_task', dataset='dots', preprocessing='min', trainBool=False,
                              path=task+"_"+model+"/", models=[modelName], electrodes=electrodes,featureExtraction=False)
            asdf.visualizePrediction(modelNames=[modelName], nrOfruns=5, filename="Visualisation_"+model+"_run"+str(run))

            goodElectrodeIndices = np.zeros(129)
            goodElectrodeIndices[electrodes-1] = 1
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap=colour),
                               filename='Configuration_'+model, alpha=0.4,
                               pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            if task == "Position":
                asdf.plotTraining(modelFileName=modelName + "_" + str(run) + ".csv", filename="Training_" + model + "_run" + str(run), columns=["Loss", "Val_Loss"])
            else:
                asdf.plotTraining(modelFileName=modelName+"_angle"+str(run)+".csv", filename="Training_"+model+"_angle"+str(run), columns=["Loss","Accuracy"])
                asdf.plotTraining(modelFileName=modelName + "_amplitude" + str(run) + ".csv",filename="Training_" + model + "_amplitude" + str(run), columns=["Loss","Val_Loss"])

        #visualize("Position_PyramidalCNN","PyramidalCNN")
        #visualize("Direction", "Xception")
        #names = ["InceptionTime_SideFronts","InceptionTime_Top2_Amplitude","InceptionTime_Top6_Amplitude","InceptionTime_Top2_Angle","InceptionTime_Top6_Angle"]
        #electrodes = [np.array([  1,  2,  3,  8,  9, 14, 21, 22, 23, 25, 26, 27, 32, 33, 38, 43,120,121, 122,123,125,128]),np.array([ 27,123]),np.array([ 27, 38, 43,120,121,123]),np.array([125,128]),np.array([  1, 32, 38,121,125,128])]

        #for i in range(len(names)):
            #for j in range(1,6):
                #visualize("Direction", names[i], j, electrodes[i], "InceptionTime", "Purples")

        #names = ["InceptionTime_SideFronts", "InceptionTime_Top2", "InceptionTime_Top6"]
        #electrodes = [np.array([1, 2, 3, 8, 9, 14, 21, 22, 23, 25, 26, 27, 32, 33, 38, 43, 120, 121, 122, 123, 125, 128]),np.array([ 1,32]),np.array([  1, 38, 32,121,125,128])]

        #for i in range(len(names)):
            #for j in range(1,6):
                #visualize("Position",names[i],j,electrodes[i],"InceptionTime","Purples")





    if not local:

        def train(filename,network,electrodes,prep,task,trainBool=True):
            asdf = AnalEyeZor(task=task, dataset='dots', preprocessing=prep, trainBool=trainBool
                               ,models=network, electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
            del asdf


        def PFI(filename,network,electrodes,prep,task,trail=False, trainBool=True):
            asdf = AnalEyeZor(task=task, dataset='dots', preprocessing=prep, trainBool=trainBool
                               ,path=filename,models=network, electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
            if trail:
                asdf.PFI(saveTrail="_amplitude",nameSuffix="_"+network[0]+"_amplitude")
                asdf.PFI(saveTrail="_angle",nameSuffix="_"+network[0]+"_angle")
            else:
                asdf.PFI(nameSuffix="_"+network[0])
            del asdf

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

        top2Amp = np.array([1,32])
        top2Ang = np.array([125,128])
        top4 = np.array([1,32,125,128])
        top7 = np.array([1,17,32,38,121,125,128])
        sideFronts = np.array([1,2,3,8,9,14,17,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128])

        #train('Direction_Top2Amp', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top2Amp, 'min',"Direction_task")
        train('Direction_Top2Ang', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top2Ang, 'min', "Direction_task")
        #train('Direction_Top4', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top4, 'min', "Direction_task")
        #train('Direction_Top7', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], top7, 'min', "Direction_task")
        #train('Direction_SideFronts', ["InceptionTime","EEGNet","CNN","PyramidalCNN","Xception"], sideFronts, 'min', "Direction_task")


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
