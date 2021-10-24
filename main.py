import sys
import time
import logging
from config import config, create_folder
from utils.tables_utils import print_table
from Joels_Files.AnalEyeZor import AnalEyeZor
import pandas as pd
import numpy as np

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

"""
Main entry of the program
Creates the logging files, loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""

def main():
    # Setting up logging

    #asdf = AnalEyeZor(task='LR_task',dataset='antisaccade',preprocessing='max', trainBool=False, path="/Users/Hullimulli/Documents/ETH/SA2/run1/",models=["InceptionTime"],featureExtraction=True)



    #asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='min', trainBool=False, models=["InceptionTime"],featureExtraction=False)

    asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='min', trainBool=False, path="InceptionTimeLRMin/",models=["InceptionTime"],featureExtraction=False)
    lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_MinLR.csv', usecols=['InceptionTime']).to_numpy()
    goodElectrodeIndices = np.zeros(np.squeeze(lossValues).shape)
    goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:32]] = 1
    asdf.electrodePlot(colourValues=asdf.colourCode(values=np.squeeze(lossValues)),alpha=1)
    asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices), name='best32Electrode.png', alpha=0.4)
    asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='max', trainBool=False, path="InceptionTimeLRMax/",models=["InceptionTime"],featureExtraction=False)
    lossValues = pd.read_csv(asdf.currentFolderPath + 'PFI_MaxLR.csv', usecols=['InceptionTime']).to_numpy()
    goodElectrodeIndices = np.zeros(np.squeeze(lossValues).shape)
    goodElectrodeIndices[np.argsort(-np.squeeze(lossValues))[:64]] = 1
    asdf.electrodePlot(colourValues=asdf.colourCode(values=np.squeeze(lossValues)),alpha=1)
    asdf.electrodePlot(colourValues=asdf.colourCode(values=goodElectrodeIndices), name='best64Electrode.png',alpha=0.4)

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
