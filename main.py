import sys
import time
import logging
from config import config, create_folder
from utils.tables_utils import print_table
from Joels_Files.AnalEyeZor import AnalEyeZor

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

    asdf = AnalEyeZor(task='LR_task',dataset='antisaccade',preprocessing='min', models=["InceptionTime"],featureExtraction=False)
    asdf = AnalEyeZor(task='LR_task', dataset='antisaccade', preprocessing='max', models=["InceptionTime"],featureExtraction=False)

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
