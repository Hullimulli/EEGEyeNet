import numpy as np
import os
import glob
from config import config
from tqdm import tqdm
from DL_Models.tf_models.utils.losses import angle_loss
from sklearn.metrics import mean_squared_error, log_loss
from hyperparameters import batch_size

def modelPathsFromBenchmark(experimentFolderPath: str, architectures: list) -> list:
    #Check
    if not os.path.isdir(experimentFolderPath):
        raise Exception("Directory does not exist.")
    pathList = list()
    checkpointPath = os.path.join(experimentFolderPath,"checkpoint")
    runs = [x for x in os.listdir(checkpointPath) if os.path.isdir(os.path.join(checkpointPath,x))]
    for k in runs:
        models = [x for x in os.listdir(os.path.join(checkpointPath,k)) if os.path.isdir(os.path.join(checkpointPath,k,x))]
        for i in models:
            for j in architectures:
                if i.lower().startswith(j.lower()):
                    pathList.append(os.path.join(checkpointPath,k,i))
    return pathList

def PFI(inputSignals: np.ndarray, groundTruth: np.ndarray, modelPaths: list, loss: str, directory: str,
        filename: str = "PFI", iterations: int = 5) -> np.ndarray:
    if config['framework'] == 'tensorflow':
        return PFITensorflow(inputSignals = inputSignals, groundTruth = groundTruth, modelPaths=modelPaths, loss=loss, directory = directory, filename = filename,iterations = iterations)
    elif config['framework'] == 'pytorch':
        return PFITensorflow(inputSignals = inputSignals, groundTruth = groundTruth, modelPaths=modelPaths, loss=loss, directory = directory, filename = filename,iterations = iterations)
    else:
        raise Exception("PFI not available for framework.")

def PFITensorflow(inputSignals: np.ndarray, groundTruth: np.ndarray, modelPaths: list, loss: str, directory: str,
                  filename: str = "PFI", iterations: int = 5) -> np.ndarray:
    import tensorflow.keras as keras

    #Checks
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")
    for modelPath in modelPaths:
        if not os.path.isdir(modelPath):
            raise Exception("Model path {} does not exist.".format(modelPath))


    base = 0
    print("Evaluating Base.")

    for modelPath in tqdm(modelPaths):
        model = keras.models.load_model(modelPath, compile=False)
        prediction = model(inputSignals)
        if loss == "angle-loss":
            base += angle_loss(np.squeeze(groundTruth), np.squeeze(prediction))
        elif loss == 'bce':
            base += log_loss(np.squeeze(groundTruth), np.squeeze(prediction), normalize=True, eps=1e-6)
        elif loss == 'mse':
            base += mean_squared_error(np.squeeze(groundTruth), np.squeeze(prediction))
        else:
            raise Exception("Error function not yet implemented.")
    base = base / len(modelPaths)
    electrodeLosses = np.zeros(inputSignals.shape[2])

    print("Evaluating PFI.")
    for j in tqdm(range(inputSignals.shape[2])):
        for i in range(iterations):
            inputSignalsShuffled = inputSignals.copy()
            np.random.shuffle(inputSignalsShuffled[:, :, j])
            for modelPath in modelPaths:
                model = keras.models.load_model(modelPath, compile=False)
                prediction = model.predict(inputSignalsShuffled,batch_size=batch_size)
                if loss == "angle-loss":
                    electrodeLosses[j] += angle_loss(np.squeeze(groundTruth), np.squeeze(prediction))
                elif loss == 'bce':
                    electrodeLosses[j] += log_loss(np.squeeze(groundTruth), np.squeeze(prediction), normalize=True, eps=1e-6)
                elif loss == 'mse':
                    electrodeLosses[j] += mean_squared_error(np.squeeze(groundTruth), np.squeeze(prediction))
                else:
                    raise Exception("Error function not yet implemented.")
    lossRatio = np.divide((electrodeLosses / (iterations * len(modelPaths))), base) - 1
    csvTable = np.zeros([lossRatio.shape[0],2])
    csvTable[:,1] = lossRatio
    csvTable[:,0] = np.arange(lossRatio.shape[0])+1
    np.savetxt(os.path.join(directory,filename + '.csv'), csvTable, fmt='%s', delimiter=',',
               header='Electrode Number,Avg. Loss Ratio', comments='')
    return lossRatio

def PFITorch(inputSignals: np.ndarray, groundTruth: np.ndarray, modelPaths: list, loss: str, iterations: int = 5) -> np.ndarray:

    #TODO

    pass