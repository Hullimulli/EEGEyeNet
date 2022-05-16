import numpy as np
import os
import glob
from config import config
from tqdm import tqdm
from DL_Models.tf_models.utils.losses import angle_loss
from sklearn.metrics import mean_squared_error, log_loss
from hyperparameters import batch_size
from ..plotFunctions.attention_visualisations import saliencyMap

def modelPathsFromBenchmark(experimentFolderPath: str, architectures: list, angleArchitectureBool: bool = False) -> list:
    """
    Benchmark() generates an experiment directory. With this function, all save paths of desired network architectures
    are returned.

    @param experimentFolderPath: The directory in which checkpoint can be found.
    @type experimentFolderPath: String
    @param architectures: List containing all wanted architectures. The savefolder must contain this string.
    @type architectures: List of Strings
    @param angleArchitectureBool: If True, returns the network for the angle task.
    @type angleArchitectureBool: Bool
    @return: List containing the found paths
    @rtype: List of Strings
    """

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
                if angleArchitectureBool and i.lower().startswith('_angle'+j.lower()):
                    pathList.append(os.path.join(checkpointPath, k, i))
                elif not angleArchitectureBool and i.lower().startswith('_amplitude'+j.lower()):
                    pathList.append(os.path.join(checkpointPath, k, i))
    return pathList

def PFI(inputSignals: np.ndarray, groundTruth: np.ndarray, modelPaths: list, loss: str, directory: str,
        filename: str = "PFI", iterations: int = 5) -> np.ndarray:
    """
    By comparing the base loss with a loss for a dataset in which for each electrode the signals are randomly
    exchanged between all samples, we can get a metric on how important an electrode is for prediction.
    This function goes trough this procedure.

    @param inputSignals: 3d Tensor of the signals which have to be plotted. Shape has to be [#samples,#timestamps,
    #electrodes], note that for EEGEyeNet the input shape is different. This problem will be fixed in a future update.
    @type inputSignals: Numpy Array
    @param groundTruth: Ground truth label for each sample.
    @type groundTruth: Numpy Array
    @param modelPaths: A list of paths which lead to the model which will be loaded directly with the framework.
    @type modelPaths: List of Strings
    @param loss: The loss which is used.
    @type loss: String
    @param directory: Name of the directory where the .csv will be saved.
    @type directory: String
    @param filename: The name of the .csv containing the loss ratios.
    @type filename: String
    @param iterations: How often the procedures is repeated for one network.
    @type iterations: Int
    @return: Loss ratios
    @rtype: Numpy Array
    """
    if config['framework'] == 'tensorflow':
        return PFITensorflow(inputSignals = inputSignals, groundTruth = groundTruth, modelPaths=modelPaths, loss=loss, directory = directory, filename = filename,iterations = iterations)
    elif config['framework'] == 'pytorch':
        raise Exception("Pytorch is not yet implemented.")
        #return PFITorch(inputSignals = inputSignals, groundTruth = groundTruth, modelPaths=modelPaths, loss=loss, directory = directory, filename = filename,iterations = iterations)
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
        prediction = model.predict(inputSignals,batch_size=batch_size)
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

def PFITorch(inputSignals: np.ndarray, groundTruth: np.ndarray, modelPaths: list, loss: str, directory: str,
             filename: str = "PFI", iterations: int = 5) -> np.ndarray:

    import torch

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for modelPath in tqdm(modelPaths):
        model = torch.load(modelPath)
        model.eval()
        prediction = model.predict(inputSignals,batch_size=batch_size)
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
                model = torch.load(modelPath)
                model.eval()
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


def gradientPFI(inputSignals: np.ndarray, groundTruth: np.ndarray, modelPaths: list, loss: str, directory: str,
                  filename: str = "PFI") -> np.ndarray:

    #Checks
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")
    for modelPath in modelPaths:
        if not os.path.isdir(modelPath):
            raise Exception("Model path {} does not exist.".format(modelPath))


    base = np.zeros([inputSignals.shape[2]])
    print("Evaluating PFI.")
    if config['framework'] == 'tensorflow':
        import tensorflow.keras as keras

    for modelPath in tqdm(modelPaths):
        if config['framework'] == 'tensorflow':
            model = keras.models.load_model(modelPath, compile=False)
        baseModel = np.nanmean(saliencyMap(model=model,loss=loss,inputSignals=inputSignals,groundTruth=groundTruth),axis=0,keepdims=True)
        baseModel = np.nanmean(baseModel,axis=1)
        base += np.squeeze(baseModel)

    base = base / len(modelPaths)

    csvTable = np.zeros([base.shape[0],2])
    csvTable[:,1] = base
    csvTable[:,0] = np.arange(base.shape[0])+1
    np.savetxt(os.path.join(directory,filename + '.csv'), csvTable, fmt='%s', delimiter=',',
               header='Electrode Number,Avg. Gradient', comments='')
    return base
