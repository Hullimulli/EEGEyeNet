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

        def plot(filename):
            asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='min', trainBool=False,
                             path="LRMin_InceptionTime_All/", models=["InceptionTime"], featureExtraction=False)
            lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_'+filename+'.csv', usecols=['InceptionTime']).to_numpy()
            goodElectrodeIndices = np.zeros(np.squeeze(lossValues).shape)
            #goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:1]] = 1
            goodElectrodeIndices[0] = 1
            asdf.topoPlot(lossValues,pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            #asdf.electrodePlot(colourValues=asdf.colourCode(values=np.squeeze(lossValues),colour="blue"),name='Electrode_Configuration_'+filename+'.png',alpha=1, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            #asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices,colour="blue"), name='best2Electrode_'+filename+'.png', alpha=0.4, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")
            #goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:4]] = 1
            #asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices, colour="blue"),
            #                  name='best4Electrode_'+filename+'.png', alpha=0.4, pathForOriginalRelativeToExecutable="./Joels_Files/forPlot/")

        def train(filename,electrodes,prep):
            asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing=prep, trainBool=True
                              , models=["InceptionTime"], electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,modelName=["InceptionTime"],originalPath=asdf.currentFolderPath)

        plot("Min_BCLoss_IncreaseInPercent")




    if not local:

        def train(filename,network,electrodes,prep,task,trail=False, trainBool=True):
            asdf = AnalEyeZor(task=task, dataset='dots', preprocessing=prep, trainBool=trainBool
                               ,path="Direction_Xception_PFI/",models=network, electrodes=electrodes,featureExtraction=False)
            asdf.moveModels(newFolderName=filename,originalPath=asdf.currentFolderPath)
            if trail:
                #asdf.PFI(saveTrail='angle', nameSuffix='angle')
                asdf.PFI(saveTrail='amplitude', nameSuffix='amplitude')
            else:
                asdf.PFI()
            del asdf

        #train('LRMin_InceptionTime_Top2',np.array([1,32]),'min')
        #train('LRMin_InceptionTime_Top6', np.array([1, 32, 38, 121, 125, 128]), 'min')
        #train('LRMin_InceptionTime_Front', np.array([43, 38, 33, 128, 32, 25, 21, 17, 14, 8, 1, 125, 122, 121, 120]), 'min')
        #train('LRMin_InceptionTime_All', 1 + np.arange(129), 'min')
        train('Direction_Xception_PFI', ["Xception"],1 + np.arange(129), 'min', "Direction_task",trail=True, trainBool=False)
        train('Direction_PyramidalCNN_PFI', ["PyramidalCNN"], 1 + np.arange(129), 'min', "Direction_task",trail=True)
        train('Position_PyramidalCNN_PFI', ["PyramidalCNN"], 1 + np.arange(129), 'min', "Position_task")

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
