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
            asdf = AnalEyeZor(task='Position_task', dataset='dots', preprocessing='min', trainBool=False,
                             path="Position_All/", models=[model], featureExtraction=False)
            lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_'+filename+'.csv', usecols=[model]).to_numpy()
            goodElectrodeIndices = np.zeros(np.squeeze(lossValues).shape)
            goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:2]] = 1
            asdf.electrodeBarPlot(values=lossValues, colour='purple',name="Electrode_Loss_"+filename)
            asdf.topoPlot(lossValues,cmap=colour,filename="Topoplot_"+filename,pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/",epsilon=0.01)
            asdf.electrodePlot(colourValues=asdf.colourCode(values=np.squeeze(lossValues),colourMap=colour,epsilon=0.01),filename='Electrode_Losses_'+filename,alpha=1, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices,colourMap=colour), filename='best2Electrode_'+filename, alpha=0.4, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:4]] = 1
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap=colour),
                              filename='best4Electrode_'+filename, alpha=0.4, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        plot("EEGNet","EEGNet","Purples")
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
                              path="Position_All/", models=["CNN"], featureExtraction=False)
            asdf.combineResults("statistics.csv",directories,filename="Position_CNN_Statistics",nameStartIndex=9,addNrOfParams=True)
            asdf.generateTable("Position_CNN_Statistics.csv",filename='Position_CNN_Statistics',addNrOfParams=False)

        #directories = ["Direction_CNN_SideFronts/", "Direction_CNN_Top2_Amplitude/", "Direction_CNN_Top6_Amplitude/", "Direction_CNN_Top2_Angle/", "Direction_CNN_Top6_Angle/"]
        #directories = ["Position_CNN_SideFronts/", "Position_CNN_Top2/", "Position_CNN_Top6/"]
        #combineResults(directories)

        def visualize(task, model,run,electrodes):
            asdf = AnalEyeZor(task=task+'_task', dataset='dots', preprocessing='min', trainBool=False,
                              path=task+"_"+model+"/", models=["CNN"], electrodes=electrodes,featureExtraction=False)
            asdf.visualizePrediction(modelName="CNN", run=run, filename="Visualisation_"+model+"_run"+str(run))

            goodElectrodeIndices = np.zeros(129)
            goodElectrodeIndices[electrodes-1] = 1
            asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colourMap='Blues'),
                               filename='Configuration', alpha=0.4,
                               pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        #visualize("Position_PyramidalCNN","PyramidalCNN")
        #visualize("Direction", "Xception")
        #names = ["CNN_SideFronts","CNN_Top2_Amplitude","CNN_Top6_Amplitude","CNN_Top2_Angle","CNN_Top6_Angle"]
        #electrodes = [np.array([  1,  2,  3,  8,  9, 14, 21, 22, 23, 25, 26, 27, 32, 33, 38, 43,120,121, 122,123,125,128]),np.array([ 27,123]),np.array([  3, 23, 27, 43,120,123]),np.array([125,128]),np.array([  1,  2, 26, 32,125,128])]

        #names = ["CNN_SideFronts", "CNN_Top2", "CNN_Top6"]
        #electrodes = [np.array([1, 2, 3, 8, 9, 14, 21, 22, 23, 25, 26, 27, 32, 33, 38, 43, 120, 121, 122, 123, 125, 128]),np.array([ 1,32]),np.array([  1, 32, 38,121,125,128])]

        #for i in range(len(names)):
            #for j in range(1,6):
                #visualize("Position",names[i],j,electrodes[i])
                #visualizeTraining("Position", "CNN"+"_"+str(j)+".csv",names[i],["Loss","Val_Loss"])
                #visualizeTraining("Direction", "CNN"+"_angle"+str(j)+".csv", names[i],["Loss","Accuracy"])






    if not local:

        def train(filename,network,electrodes,prep,task,trainBool=True):
            asdf = AnalEyeZor(task=task, dataset='dots', preprocessing=prep, trainBool=trainBool
                               ,models=network, electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
            del asdf


        def PFI(filename,network,electrodes,prep,task,trail=False, trainBool=True):
            asdf = AnalEyeZor(task=task, dataset='dots', preprocessing=prep, trainBool=trainBool
                               ,path=filename,models=network, electrodes=electrodes,featureExtraction=False)
            #asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
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

        train('Direction_EEGNet_Top2_Amplitude', ["EEGNet"], np.array([38,121]), 'min', "Direction_task")
        train('Direction_EEGNet_Top2_Angle', ["EEGNet"], np.array([1,32]), 'min', "Direction_task")
        train('Direction_EEGNet_Top6_Angle', ["EEGNet"], np.array([1,27,32,123,125,128]), 'min', "Direction_task")
        train('Direction_EEGNet_Top6_Amplitude', ["EEGNet"], np.array([27,38,43,120,121,123]), 'min', "Direction_task")
        train('Direction_EEGNet_SideFronts', ["EEGNet"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")

        #train('Direction_CNN_Top2_Amplitude', ["CNN"], np.array([27,123]), 'min', "Direction_task")
        #train('Direction_CNN_Top2_Angle', ["CNN"], np.array([125,128]), 'min', "Direction_task")
        #train('Direction_CNN_Top6_Angle', ["CNN"], np.array([1,2,26,32,125,128]), 'min', "Direction_task")
        #train('Direction_CNN_Top6_Amplitude', ["CNN"], np.array([3,23,27,43,120,123]), 'min', "Direction_task")
        #train('Direction_CNN_SideFronts', ["CNN"], np.array([1,2,3,8,9,14,21,22,23,25,26,27,32,33,38,43,120,121,122,123,125,128]), 'min', "Direction_task")

        #train('Position_All', ["Xception","CNN","PyramidalCNN","InceptionTime","EEGNet"], 1 + np.arange(129), 'min', "Position_task", trail=True,trainBool=True)

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
