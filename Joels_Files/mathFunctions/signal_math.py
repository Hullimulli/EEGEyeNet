from tqdm import tqdm
import os
import numpy as np
from config import config

def pca(inputSignals: np.ndarray, filename: str, directory: str):
    """
    Calculates the principle components for the given samples. Saves the matrix as a numpy file.

    @param inputSignals: 3d Tensor of the signals of which the principle components have to be calculated.
    Shape has to be [#samples,#timestamps,#electrodes]
    @type inputSignals: Numpy Array
    @param filename: The name of the file which stores the eigenvectors.
    @type filename: String
    @param directory: Directory where the file has to be saved.
    @type directory: String
    """
    #Checks
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    if not os.path.isdir(directory):
        raise Exception("Directory does not exist.")

    eigenVectors = np.zeros([inputSignals.shape[1] ,inputSignals.shape[1] ,inputSignals.shape[2]])
    for i in tqdm(range(inputSignals.shape[2])):
        covMatrix = np.cov(np.transpose(inputSignals[:, :, i]))
        if config['framework'] == 'tensorflow':
            import tensorflow as tf
            e, v = tf.linalg.eigh(covMatrix)
        elif config['framework'] == 'pytorch':
            import torch
            e, v = torch.linalg.eigh(torch.from_numpy(covMatrix))
        else:
            print("No valid framework selected.")
            return
        del e
        eigenVectors[: ,: ,i] = v.numpy()[: ,::-1]
    np.save(os.path.join(directory,filename) ,eigenVectors)


def pcaDimReduction(inputSignals, file, dim=2, transformBackBool = True):
    """
    Takes the signals and transforms them with the eigenvector matrix of the PCA. All values except the first n ones,
    which correspond to the eigenvectors with the highest eigenvalues, are set to zero. Then the
    signal is transformed back to its original space if desired.

    @param inputSignals: 3d Tensor of the signal of which the principle components have to be calculated.
    Shape has to be [#samples,#timestamps,#electrodes]
    @type inputSignals: Numpy Array
    @param file: Where the numpy file of the principle components is found.
    @type file: String
    @param dim: How many eigenvectors are kept.
    @type dim: Integer
    @param transformBack: If True, the data is transformed back to its original space.
    @type transformBack: Bool
    @return: Input transformed with PCA.
    @rtype: [#samples,#timestamps/#dimensions,#electrodes] Numpy Array
    """

    #Checks
    if inputSignals.ndim != 3:
        raise Exception("Need a 3 dimensional array as input.")
    if not os.path.isfile(file):
        raise Exception("Directory does not exist.")
    v = np.load(file)

    if inputSignals.shape[1] != v.shape[1] or inputSignals.shape[2] != v.shape[2]:
        raise Exception("Invalid shapes.")
    if config['framework'] == 'tensorflow':
        import tensorflow as fr
        z = np.transpose(fr.matmul(np.transpose(v), np.transpose(inputSignals)).numpy())
        if transformBackBool:
            z[:, dim:, :] = 0
            returnValue = np.transpose(fr.matmul(np.swapaxes(np.transpose(v), 1, 2), np.transpose(z)).numpy())
            return returnValue
        else:
            return z[:, :dim, :]
    elif config['framework'] == 'pytorch':
        import torch as fr
        z = np.transpose(fr.matmul(fr.from_numpy(np.transpose(v)), fr.from_numpy(np.transpose(inputSignals))).numpy())
        if transformBackBool:
            z[:, dim:, :] = 0
            returnValue = np.transpose(fr.matmul(fr.from_numpy(np.swapaxes(np.transpose(v), 1, 2)),
                                                 fr.from_numpy(np.transpose(z))).numpy())
            return returnValue
        else:
            return z[:, :dim, :]
    else:
        raise Exception("No valid framework selected.")