import sys
import time
import logging
from config import config, create_folder
from utils.tables_utils import print_table
from Joels_Files.AnalEyeZor import AnalEyeZor
import pandas as pd
import numpy as np
from benchmark import benchmark, split
from utils import IOHelper
from tqdm import tqdm
from Joels_Files.mathFunctions import electrode_math
from ninaExperiment import experiment
from Joels_Files.TwoDArchitecture.Train import method

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

        ORI_ORDER = [i for i in range(129)]

        NEW_ORDER_1 = list(np.arange(73, 125 + 1)) + list(np.arange(1, 7 + 1)) + [129] + \
                      list(np.arange(8, 16 + 1)) + list(np.arange(18, 20 + 1)) + [126, 17] + \
                      list(np.arange(21, 24 + 1)) + [127] + list(np.arange(25, 31 + 1)) + [128] + \
                      list(np.arange(32, 72 + 1))
        NEW_ORDER_1 = [i - 1 for i in NEW_ORDER_1]
        assert len(NEW_ORDER_1) == 129




        trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
        trainX.transpose((0,2,1))
        start_time = time.time()
        print('Starting the experiment...')

        TOP4 = [1, 32, 125, 128]
        TOP7 = TOP4 + [17, 38, 121]
        SIDE_FRONTS = TOP7 + [2, 3, 8, 9, 14, 21, 22, 23, 25, 26, 27, 33, 43, 120, 122, 123]
        electrode_chose = None  # SIDE_FRONTS
        if electrode_chose is not None:
            index = [each - 1 for each in electrode_chose]
            trainX = trainX[:, :, index]
            print('Choose Electrodes: {}'.format(electrode_chose))

        timeframe_chose = None  # (200,300) #None
        if timeframe_chose != None:
            trainX = trainX[:, timeframe_chose[0]:timeframe_chose[1], :]
            print('Choose Timeframe: {}'.format(timeframe_chose))

        change_electrode_order = False
        if change_electrode_order:
            new_order = NEW_ORDER_1
            trainX = trainX[:, :, new_order]
            print('Change Electrode Order into: {}'.format(new_order))

        experiment(trainX, trainY)
        print('Finish the experiment.')

        print("--- Runtime: %s seconds ---" % (time.time() - start_time))






    if not local:
        task = method(directory='./MultiDNet', nrOfEpochs=10, wandbProject='eegeye')
        #task = method(directory='/Users/Hullimulli/Documents/ETH/SA2/dev_EEGEyeNet', nrOfEpochs=10, wandbProject='',batchSize=8)
        task.fit()
        def PFINew():
            pathlist = electrode_math.modelPathsFromBenchmark(
                "/home/kjoel/SA/runs/1651674768_Direction_task_dots_min",
                ["PyramidalCNN","CNN","InceptionTime","Xception"],angleArchitectureBool=False)
            trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1]
            trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0]
            ids = trainY[:, 0]
            trainIndices, valIndices, testIndices = split(ids, 0.7, 0.15, 0.15)
            trainY = trainY[:,1]
            losses = electrode_math.PFI(inputSignals=trainX[valIndices], groundTruth=trainY[valIndices], loss='mse', directory="./",filename="PFI_Amplitude",
                                        modelPaths=pathlist, iterations=5)

        #PFINew()

if __name__=='__main__':
    main()
