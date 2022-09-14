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


def convertToImage(dataX: np.ndarray,normalizeBool: bool = False):
    electrodePositions = getElectrodeIndices()
    map = np.zeros([dataX.shape[0],32,32,dataX.shape[1]],dtype=np.float32)
    for j,i in enumerate(electrodePositions):
        map[:,i[1],i[0],:] = dataX[:,:,j]

    if normalizeBool:
         map = map - np.min(map,axis=(1,2,3),keepdims=True)
         map = map / np.max(map,axis=(1,2,3),keepdims=True)
    return map


def convertToVideo(dataX: np.ndarray,normalizeBool: bool = True):
    electrodePositions = getElectrodeIndices()
    map = np.zeros([dataX.shape[0],32,32,dataX.shape[1],1],dtype=np.float32)
    for j,i in enumerate(electrodePositions):
        map[:,i[1],i[0],:,0] = dataX[:,:,j]

    if normalizeBool:
         map = map - np.min(map,axis=(1,2,3),keepdims=True)
         map = map / np.max(map,axis=(1,2,3),keepdims=True)
    return map

def checkImageConfiguration():
    pathOfFile = os.path.join(Path(__file__).resolve().parent, "filesForMath")
    electrodePositions = sio.loadmat(os.path.join(pathOfFile, "lay129_head.mat"))['lay129_head']['pos'][0][0]

    electrodePositions = 37.5*electrodePositions[3:132]
    #37.5


    plt.scatter(electrodePositions[:,0],electrodePositions[:,1],label="True Position")
    electrodePositions = electrodePositions.astype(np.int)
    coordMax = np.amax(np.abs(electrodePositions), axis=0)
    print(coordMax)
    plt.scatter(electrodePositions[:, 0], electrodePositions[:, 1],label="Rounded Position")
    plt.legend()
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
