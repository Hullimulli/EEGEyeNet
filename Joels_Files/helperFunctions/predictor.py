from ..mathFunctions.electrode_math import modelPathsFromBenchmark

import tensorflow.keras as keras
import numpy as np
import os
import glob
from config import config
from tqdm import tqdm
from hyperparameters import batch_size

def savePredictions(filename: str,savePath: str ,experimentFolderPath: str, architectures: list, inputSignals: np.ndarray,angleArchitectureBool: bool = False, saveBool: bool = False):

    paths = modelPathsFromBenchmark(experimentFolderPath,architectures,angleArchitectureBool)
    nrOfLabels = 1
    if config['task'] == 'Position_task':
        nrOfLabels = 2

    predictions = np.zeros([len(architectures),int(len(paths)/len(architectures)),inputSignals.shape[0],nrOfLabels])
    predictions = np.squeeze(predictions)
    for j in tqdm(range(len(architectures))):
        for i in range(int(len(paths)/len(architectures))):
            if 'EEGNet' in paths[j*int(len(paths)/len(architectures))+i]:
                trainX = np.transpose(inputSignals, (0, 2, 1))
            else:
                trainX = inputSignals
            print("Evaluating",paths[j*int(len(paths)/len(architectures))+i])
            model = keras.models.load_model(paths[j*int(len(paths)/len(architectures))+i], compile=False)
            predictions[j,i] = np.squeeze(model.predict(trainX,batch_size=batch_size))
    if saveBool:
        np.save(os.path.join(savePath,config['task'])+"_"+filename,predictions)
    return predictions