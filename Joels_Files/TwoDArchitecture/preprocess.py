import numpy as np
import os
from pathlib import Path
import scipy.io as sio
import tensorflow as tf

import matplotlib.pyplot as plt
from config import config

def getElectrodeIndices():
    pathOfFile = os.path.join(Path(__file__).resolve().parent, "filesForMath")
    electrodePositions = sio.loadmat(os.path.join(pathOfFile, "lay129_head.mat"))['lay129_head']['pos'][0][0]

    electrodePositions = 37.5*electrodePositions[3:132]
    electrodePositions = electrodePositions.astype(np.int)
    electrodePositions[:,0] -= np.min(electrodePositions[:,0])
    electrodePositions[:, 1] -= np.min(electrodePositions[:, 1])
    return electrodePositions


def convertToImage(dataX: np.ndarray):
    electrodePositions = getElectrodeIndices()
    map = np.zeros([dataX.shape[0],32,32,dataX.shape[1]])
    for j,i in enumerate(electrodePositions):
        map[:,i[1],i[0],:] = dataX[:,:,j]

    return map

def checkImageConfiguration():
    pathOfFile = os.path.join(Path(__file__).resolve().parent, "filesForMath")
    electrodePositions = sio.loadmat(os.path.join(pathOfFile, "lay129_head.mat"))['lay129_head']['pos'][0][0]

    electrodePositions = 37.5*electrodePositions[3:132]
    #37.5


    plt.scatter(electrodePositions[:,0],electrodePositions[:,1])
    electrodePositions = electrodePositions.astype(np.int)
    coordMax = np.amax(np.abs(electrodePositions), axis=0)
    print(coordMax)
    plt.scatter(electrodePositions[:, 0], electrodePositions[:, 1])
    plt.show()
    plt.cla()
    electrodePositions = electrodePositions.astype(np.int)
    electrodePositions[:,0] -= np.min(electrodePositions[:,0])
    electrodePositions[:, 1] -= np.min(electrodePositions[:, 1])
    map = np.zeros([(1+coordMax[0])*2,(1+coordMax[1])*2])
    for j,i in enumerate(electrodePositions):
        if map[i[1],i[0]] != 0:
            print('Already has Value for {}'.format(j+1))
        map[i[1],i[0]] = j
    plt.imshow(map,cmap="jet")
    plt.show()
