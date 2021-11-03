import sys
import os
import time
import logging
import math
from config import config, create_folder
from utils import IOHelper
from benchmark import benchmark, split
from utils.tables_utils import print_table
from hyperparameters import our_DL_models, our_ML_models, your_models, all_models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf
import shutil
import pandas as pd
import re
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import mne

import numpy as np


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

class AnalEyeZor():

    def __init__(self, task, dataset, preprocessing, models, electrodes = 1+np.arange(129),featureExtraction = False, trainBool = True, saveModelBool = True, path=None):

        config['include_ML_models'] = False
        config['include_DL_models'] = False
        config['include_your_models'] = True
        config['include_dummy_models'] = False
        config['feature_extraction'] = featureExtraction
        self.electrodes = electrodes
        self.inputShape = (1, 258) if config['feature_extraction'] else (500, electrodes.shape[0])
        self.numberOfNetworks = 5
        self.currentFolderPath = ""
        config['task'] = task
        config['dataset'] = dataset
        config['preprocessing'] = preprocessing

        def build_file_name():
            all_EEG_file = config['task'] + '_with_' + config['dataset']
            all_EEG_file = all_EEG_file + '_' + 'synchronised_' + config['preprocessing']
            all_EEG_file = all_EEG_file + ('_hilbert.npz' if config['feature_extraction'] else '.npz')
            return all_EEG_file

        config['all_EEG_file'] = build_file_name()
        asdf = our_DL_models[config['task']][config['dataset']][config['preprocessing']]
        #########################################Loading_or_Training_Model####################################################
        your_models[config['task']] = {config['dataset']:{config['preprocessing']:{}}}
        for i in models:
            if config['task'] == 'Direction_task':
                if i not in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] and i not in our_ML_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] \
                and i not in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['angle'] and i not in our_ML_models[config['task']][config['dataset']][config['preprocessing']]['angle']:
                    print("{} not yet configured.".format(i))
                else:
                    if i in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] and i in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['angle']:
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] = {i:{}}
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][i] = \
                        our_DL_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][i]
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'] = {i: {}}
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][i] = \
                        our_DL_models[config['task']][config['dataset']][config['preprocessing']]['angle'][i]
                    else:
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] = {i:{}}
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][i] = \
                        our_ML_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][i]
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'] = {i: {}}
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][i] = \
                        our_ML_models[config['task']][config['dataset']][config['preprocessing']]['angle'][i]
                    your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][i][1]["input_shape"] = self.inputShape
                    your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][i][1]["input_shape"] = self.inputShape
            else:

                if i not in our_DL_models[config['task']][config['dataset']][config['preprocessing']] and i not in our_ML_models[config['task']][config['dataset']][config['preprocessing']]:
                    print("{} not yet configured.".format(i))
                else:
                    if i in our_DL_models[config['task']][config['dataset']][config['preprocessing']]:
                        your_models[config['task']][config['dataset']][config['preprocessing']] = {i:{}}
                        your_models[config['task']][config['dataset']][config['preprocessing']][i] = \
                        our_DL_models[config['task']][config['dataset']][config['preprocessing']][i]
                    else:
                        your_models[config['task']][config['dataset']][config['preprocessing']] = {i: {}}
                        your_models[config['task']][config['dataset']][config['preprocessing']][i] = \
                        our_ML_models[config['task']][config['dataset']][config['preprocessing']][i]
                    your_models[config['task']][config['dataset']][config['preprocessing']][i][1]["input_shape"] = self.inputShape


        all_models.pop(config['task'], None)
        all_models[config['task']] = your_models[config['task']]

        def initFolder():
            create_folder()
            logging.basicConfig(filename=config['info_log'], level=logging.INFO)
            logging.info('Started the Logging')
            logging.info("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
            if os.path.exists(config['model_dir'] + '/console.out'):
                f = open(config['model_dir'] + '/console.out', 'a')
            else:
                f = open(config['model_dir'] + '/console.out', 'w')
            sys.stdout = Tee(sys.stdout, f)

        if trainBool:
            config['retrain'] = trainBool
            config['save_models'] = saveModelBool
            initFolder()
            self.currentFolderPath = config['model_dir'] + "/"
            logging.info("------------------------------Loading the Data------------------------------")
            trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
            trainX = trainX[:, :, self.electrodes.astype(np.int)-1]
            logging.info("------------------------------Calling Benchmark------------------------------")
            benchmark(trainX, trainY)
            logging.info("------------------------------Finished Training------------------------------")
        else:
            if path==None:
                print("Configure the Path, where to find the network in runs.")
                raise Exception()
            self.currentFolderPath = config['log_dir'] + path
            config['load_experiment_dir'] = path
            config['retrain'] = trainBool
            initFolder()
            logging.info("------------------------------Selected Model------------------------------")



    def PFI(self, scale = False, iterations=5, useAccuracyBool=False):
        logging.info("------------------------------PFI------------------------------")
        if config['feature_extraction'] == True:
            print("No PFI for Transformed Data")
            return

        config['retrain'] = False
        config['save_models'] = False
        dataShape = np.shape(IOHelper.get_npz_data(config['data_dir'], verbose=True)[0])


        trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
        ids = trainY[:, 0]
        trainIndices, valIndices, testIndices = split(ids, 0.7, 0.15, 0.15)
        if scale:
            logging.info('Standard Scaling')
            scaler = StandardScaler()
            scaler.fit(trainX[trainIndices])
            trainX = scaler.transform(trainX[valIndices])
        trainX, valY = trainX[valIndices], trainY[valIndices,1]
        del trainIndices, valIndices, testIndices, trainY

        modelLosses = dict()
        models = all_models[config['task']][config['dataset']][config['preprocessing']]
        for name, model in models.items():
            electrodeLosses = np.zeros(dataShape[2])
            start_time = time.time()
            offset = 0
            trainer = model[0](**model[1])
            prediction = np.zeros(valY.shape)
            print("Evaluating Base of {}".format(name))
            for i in range(self.numberOfNetworks):
                path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
                trainer.ensemble.load_file_pattern = re.compile(name + '_nb_*', re.IGNORECASE)
                trainer.load(path)
                trainer.type = 'regressor'
                prediction = np.squeeze(trainer.predict(trainX))

                if config['task'] == 'LR_task':
                    if useAccuracyBool:
                        prediction = np.rint(prediction)
                    offset += self.binaryCrossEntropyLoss(valY, prediction)
                elif config['task'] == 'Direction_task':
                    offset += self.meanSquareError(valY, prediction)
                elif config['task'] == 'Position_task':
                    offset += self.euclideanDistance(valY, prediction)
            offset = offset / self.numberOfNetworks

            print("Evaluating PFI of {}".format(name))
            for k in range(iterations):
                for j in tqdm(range(int(dataShape[2]))):
                    valX = trainX.copy()
                    np.random.shuffle(valX[:, :, j])
                    prediction = np.zeros(valY.shape)
                    for i in range(self.numberOfNetworks):
                        path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
                        trainer.ensemble.load_file_pattern = re.compile(name + '_nb_*', re.IGNORECASE)
                        trainer.load(path)
                        trainer.type = 'regressor'
                        prediction += np.squeeze(trainer.predict(valX))

                        if config['task'] == 'LR_task':
                            if useAccuracyBool:
                                prediction = np.rint(prediction)
                            electrodeLosses[j] += self.binaryCrossEntropyLoss(valY, prediction)
                        elif config['task'] == 'Direction_task':
                            electrodeLosses[j] += self.meanSquareError(valY, prediction)
                        elif config['task'] == 'Position_task':
                            electrodeLosses[j] += self.euclideanDistance(valY, prediction)

            modelLosses[name] = np.divide((electrodeLosses / (iterations*self.numberOfNetworks)), offset) - 1
            runtime = (time.time() - start_time)
            logging.info("--- Sorted Electrodes According to Influence for {}:".format(name))
            logging.info(1+(np.argsort(modelLosses[name]))[::-1])
            logging.info("--- Losses of each Electrode for {}:".format(name))
            logging.info(modelLosses[name][((np.argsort(modelLosses[name]))[::-1])])
            logging.info("--- Runtime: %s for seconds ---" % runtime)
        logging.info("------------------------------Evaluated Electrodes------------------------------")
        results = np.expand_dims(np.arange(1,list(modelLosses.values())[0].shape[0]+1),0).astype(np.int)
        legend = 'Electrode Number'
        for i,j in modelLosses.items():
            legend += ','+i
            results = np.concatenate((results,np.expand_dims(j,0)),axis=0)

        np.savetxt(config['model_dir'] +  'PFI.csv', results.transpose(), fmt='%s', delimiter=',', header=legend, comments='')
        return modelLosses

    def electrodePlot(self, colourValues, name="Electrode_Configuration.png", alpha=0.4, pathForOriginalRelativeToExecutable="./EEGEyeNet/Joels_Files/forPlot/"):
        if not os.path.exists(pathForOriginalRelativeToExecutable+'blank.png'):
            print(pathForOriginalRelativeToExecutable+'blank.png'+" does not exist.")
            return
        img = cv2.imread(pathForOriginalRelativeToExecutable+'blank.png', cv2.IMREAD_COLOR)
        overlay = img.copy()
        coord = pd.read_csv(pathForOriginalRelativeToExecutable+'coord.csv', index_col='electrode', dtype=int)
        for i in range(colourValues.shape[0]):
            pt = coord.loc[i+1]
            x, y, r = pt['posX'], pt['posY'], pt['radius']
            cv2.circle(overlay, (x, y), r, colourValues[i,:], -1)

        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.imwrite(config['model_dir']+name, img)

    def moveModels(self, newFolderName, modelName, originalPath, getEpochMetricsBool=True):
        try:
            os.mkdir(config['log_dir']+newFolderName)
        except:
            print("Folder already exists.")
            return
        if not os.path.isdir(originalPath+"checkpoint/"):
            print("Original Folder not found.")
            return

        runNames = os.listdir(originalPath+"checkpoint/")
        for runName in runNames:
            if not runName == ".DS_Store":
                networkNames = os.listdir(originalPath+"checkpoint/"+runName+"/")
                for networkName in networkNames:
                    for name in modelName:
                        if name.lower() in networkName.lower():
                            shutil.move(originalPath+"checkpoint/"+runName+"/"+networkName,config['log_dir']+newFolderName+"/checkpoint/"+runName+"/"+networkName)

        allModelNames = pd.read_csv(originalPath+"runs.csv", usecols=["Model"])
        if os.path.exists(originalPath+"console.out") and getEpochMetricsBool:
            i = 1
            with open(os.path.join(originalPath,"console.out")) as f:
                readingValuesBool = False
                metrics = np.zeros([1,5])
                currentEpoch=0
                for line in f:
                    if i > len(allModelNames):
                        break
                    if "after" in line:
                        readingValuesBool = True
                    if "Epoch" in line:
                        currentEpoch = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", line)).astype(np.float)[0]
                    if "loss" in line and readingValuesBool:
                        metricToAppend = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", line)).astype(np.float)
                        metricToAppend = np.append(currentEpoch,metricToAppend[3:])
                        metrics = np.concatenate((metrics, np.expand_dims(metricToAppend,0)), axis=0)
                    if "before" in line and readingValuesBool==True:
                        readingValuesBool = False
                        np.savetxt(config['log_dir']+newFolderName+"/"+str(allModelNames.values[i-1][0])+'_{}.csv'.format(i), metrics[1:,:], fmt='%s',
                                   delimiter=',', header='Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy', comments='')
                        i+=1
                        metrics = np.zeros([1, 5])
                if i <= len(allModelNames):
                    np.savetxt(config['log_dir'] + newFolderName + "/" + str(allModelNames.values[i-1][0]) + '_{}.csv'.format(i),
                               metrics[1:, :], fmt='%s',
                               delimiter=',', header='Loss,Accuracy,Val_Loss,Val_Accuracy', comments='')

        if os.path.exists(originalPath + "runs.csv"):
            shutil.move(originalPath+"runs.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "statistics.csv"):
            shutil.move(originalPath + "statistics.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "info.log"):
            shutil.move(originalPath + "info.log", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "config.csv"):
            shutil.move(originalPath + "config.csv", config['log_dir'] + newFolderName)

        config['load_experiment_dir'] = newFolderName
        config['model_dir'] = config['log_dir'] + config['load_experiment_dir']
        stamp = str(int(time.time()))
        config['info_log'] = config['model_dir'] + '/' + 'inference_info_' + stamp + '.log'
        config['batches_log'] = config['model_dir'] + '/' + 'inference_batches_' + stamp + '.log'

    def colourCode(self, values, electrodes=np.arange(1,130), colour="red", minValue=5, epsilon = 0.001):
        colours = np.zeros((values.shape[0],3))

        if colour=="green":
            i = 1
        elif colour=="blue":
            i = 0
        else:
            i=2
        #For Decibel, use epsilon = 1, for good colour visualisation use small epsilon > 0
        values[np.where(values < 0)] = 0
        values = 10*np.log(values + epsilon)
        originalValueSpan = np.max(values[electrodes-1]) - np.min(values[electrodes-1])
        newValueSpan = 255 - minValue
        colours[electrodes-1,i] = ((values[electrodes-1] - np.min(values[electrodes-1])) * (newValueSpan / originalValueSpan) + minValue)
        return colours

    def plotTraining(self, name, modelFileName, columns=["Loss","Accuracy","Val_Loss","Val_Accuracy"], returnPlotBool=False, format="pdf"):

        data = pd.read_csv(config['model_dir'] + modelFileName, usecols=["Epoch"]+columns).to_numpy()
        xAxis = np.arange(int(np.max(data[0])))+1
        values = np.zeros([xAxis.shape[0],data.shape[1]-1])


        fig = plt.figure()
        plt.xlabel("Epoch")
        for i in range(1,data.shape[1]):
            plt.plot(xAxis, data[:,i], label=columns[i])
        plt.legend()
        fig.savefig(config['model_dir'] + name+".{}".format(format), format=format, transparent=True)
        if returnPlotBool:
            return fig
        plt.close()

    def topoPlot(self, values, pathForOriginalRelativeToExecutable="./EEGEyeNet/Joels_Files/forPlot/", epsilon=0.001):
        electrodePositions = sio.loadmat(pathForOriginalRelativeToExecutable+"lay129_head.mat")['lay129_head']['pos'][0][0]
        values[np.where(values < 0)] = 0
        values = 10 * np.log(values + epsilon)
        fig = plt.figure()
        im, cm = mne.viz.plot_topomap(np.squeeze(values),electrodePositions[3:132,:],show=False,cmap='Reds')
        clb = fig.colorbar(im)
        if epsilon==1:
            clb.ax.set_title("Loss in Db")
        else:
            clb.ax.set_title("10x Log-Ratio-Loss with eps={}".format(epsilon))
        plt.legend()
        plt.close()

    def meanSquareError(self,y,yPred):
        return np.sqrt(mean_squared_error(y, yPred.ravel()))

    def angleError(self,y,yPred):
        np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - yPred.ravel()), np.cos(y - yPred.ravel())))))

    def euclideanDistance(self,y,yPred):
        return np.linalg.norm(y - yPred, axis=1).mean()

    def binaryCrossEntropyLoss(self,y,yPred):
        return log_loss(y,yPred, normalize=True)



