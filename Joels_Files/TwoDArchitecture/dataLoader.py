from config import config
import numpy as np
import math

def loadData(inputPath, targetPath, mmapMode = 'c'):
    trainX = np.load(inputPath, mmap_mode=mmapMode)

    trainY = np.load(targetPath, mmap_mode=mmapMode)

    return trainX,trainY


def split(ids, train, val, test):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split+val_split:])

    return np.squeeze(np.argwhere(train)), np.squeeze(np.argwhere(val)), np.squeeze(np.argwhere(test))