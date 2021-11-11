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
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.io as sio
import mne
from texttable import Texttable
from tabulate import tabulate
import latextable

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
        """
        A class, where the task and folder location is defined.
        @param task: Problem Type, can be 'Direction_task', 'Position_task' or 'LR_task'.
        @type task: String
        @param dataset: Which data set is used, can be 'antisaccade','dots' or 'processing_speed'.
        @type dataset: String
        @param preprocessing: How the data is preprossed, can be either 'min' or 'max'.
        @type preprocessing: String
        @param models: Models, which have to be loaded or trained.
        @type models: List of Strings
        @param electrodes: Which electrodes should be used for training.
        @type electrodes: Numpy Int Array
        @param featureExtraction: If hilbert transformed data is used.
        @type featureExtraction: Bool
        @param trainBool: If the models have to be newly trained.
        @type trainBool: Bool
        @param saveModelBool: If the newly trained models have to be saved.
        @type saveModelBool: Bool
        @param path: Path to an existing folder containing the networks and .csv files as generated by AnalEyeZor.
        @type path: String
        """


        config['include_ML_models'] = False
        config['include_DL_models'] = False
        config['include_your_models'] = True
        config['include_dummy_models'] = False
        config['feature_extraction'] = featureExtraction
        self.electrodes = electrodes
        self.inputShape = (1, 258) if config['feature_extraction'] else (500, electrodes.shape[0])
        self.numberOfNetworks = 5
        self.currentFolderPath = ""
        self.modelNames = models
        config['task'] = task
        config['dataset'] = dataset
        config['preprocessing'] = preprocessing

        def build_file_name():
            all_EEG_file = config['task'] + '_with_' + config['dataset']
            all_EEG_file = all_EEG_file + '_' + 'synchronised_' + config['preprocessing']
            all_EEG_file = all_EEG_file + ('_hilbert.npz' if config['feature_extraction'] else '.npz')
            return all_EEG_file

        config['all_EEG_file'] = build_file_name()
        #########################################Loading_or_Training_Model####################################################
        your_models[config['task']] = {config['dataset']:{config['preprocessing']:{}}}
        firstModel = True

        #Generating corresponding config list to be used in benchmark
        for model in models:
            if config['task'] == 'Direction_task':
                if model not in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] and model not in our_ML_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] \
                and model not in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['angle'] and model not in our_ML_models[config['task']][config['dataset']][config['preprocessing']]['angle']:
                    print("{} not yet configured.".format(model))
                else:
                    if firstModel:
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] = {model: {}}
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'] = {model: {}}
                        firstModel = False
                    else:
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][model] = {}
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][model] = {}
                    if model in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'] and model in our_DL_models[config['task']][config['dataset']][config['preprocessing']]['angle']:
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][model] = \
                        our_DL_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][model]

                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][model] = \
                        our_DL_models[config['task']][config['dataset']][config['preprocessing']]['angle'][model]
                    else:
                        your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][model] = \
                        our_ML_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][model]
                        your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][model] = \
                        our_ML_models[config['task']][config['dataset']][config['preprocessing']]['angle'][model]
                    your_models[config['task']][config['dataset']][config['preprocessing']]['angle'][model][1]["input_shape"] = self.inputShape
                    your_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][model][1]["input_shape"] = self.inputShape
            else:

                if model not in our_DL_models[config['task']][config['dataset']][config['preprocessing']] and model not in our_ML_models[config['task']][config['dataset']][config['preprocessing']]:
                    print("{} not yet configured.".format(model))
                else:
                    if firstModel:
                        your_models[config['task']][config['dataset']][config['preprocessing']] = {
                            model: {}}
                        firstModel = False
                    else:
                        your_models[config['task']][config['dataset']][config['preprocessing']][model] = {}

                    if model in our_DL_models[config['task']][config['dataset']][config['preprocessing']]:
                        your_models[config['task']][config['dataset']][config['preprocessing']][model] = \
                        our_DL_models[config['task']][config['dataset']][config['preprocessing']][model]
                    else:
                        your_models[config['task']][config['dataset']][config['preprocessing']][model] = \
                        our_ML_models[config['task']][config['dataset']][config['preprocessing']][model]
                    your_models[config['task']][config['dataset']][config['preprocessing']][model][1]["input_shape"] = self.inputShape

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
            trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:, :, self.electrodes.astype(np.int)-1]
            trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1]
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

    def PFI(self, scale = False, nameSuffix='',iterations=5, useAccuracyBool=False, saveTrail=''):
        """
        Function that uses the folder which is defined when initialised to perform PFI with all contained networks.
        @param scale: Set to true, if the input has to be scaled.
        @type scale: Bool
        @param iterations: How many times the PFI is done. Averages over all iterations.
        @type iterations: Int
        @param useAccuracyBool: If True, uses accuracy instead of bce for LR-Task scoring.
        @type useAccuracyBool: Bool
        @param saveTrail: The string, with which the save-folder begins before stating the networks name. Should be either '', '_amplitude' or '_angle'
        @type saveTrail: String
        @return: Returns the ratio of loss between the prediction when one electrode is permutated and when nothing is permutated.
        This is done for each electrode. Electrode number is the index + 1.
        @rtype: Dictionary with numpy array items
        """
        logging.info("------------------------------PFI------------------------------")
        if config['feature_extraction'] == True:
            print("No PFI for Transformed Data")
            return

        config['retrain'] = False
        config['save_models'] = False

        trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
        dataShape = np.shape(trainX)
        ids = trainY[:, 0]
        trainIndices, valIndices, testIndices = split(ids, 0.7, 0.15, 0.15)
        if scale:
            logging.info('Standard Scaling')
            scaler = StandardScaler()
            scaler.fit(trainX[trainIndices])
            trainX = scaler.transform(trainX[valIndices])
        trainX, valY = trainX[valIndices], trainY[valIndices,1:]
        del trainIndices, valIndices, testIndices, trainY

        modelLosses = dict()
        if saveTrail == '_angle':
            models = all_models[config['task']][config['dataset']][config['preprocessing']]['angle']
            valY = valY[:, 1]
        elif saveTrail == '_amplitude':
            models = all_models[config['task']][config['dataset']][config['preprocessing']]['amplitude']
            valY = valY[:, 0]
        else:
            models = all_models[config['task']][config['dataset']][config['preprocessing']]

        for name, model in models.items():
            electrodeLosses = np.zeros(dataShape[2])
            start_time = time.time()
            offset = 0
            trainer = model[0](**model[1])
            print("Evaluating Base of {}".format(name))
            for i in range(self.numberOfNetworks):
                path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
                #Modification in benchmark.py st. an angle network does not overwrite amplitude
                trainer.ensemble.load_file_pattern = re.compile(saveTrail + name + '_nb_*', re.IGNORECASE)
                trainer.load(path)
                #This is a work around in order to get the non rounded data of a classifier.
                trainer.type = 'regressor'
                prediction = np.squeeze(trainer.predict(trainX))

                if config['task'] == 'LR_task':
                    if useAccuracyBool:
                        prediction = np.rint(prediction)
                    offset += self.binaryCrossEntropyLoss(valY, prediction)
                elif config['task'] == 'Direction_task' and saveTrail == '_amplitude':
                    offset += self.meanSquareError(valY, prediction)
                elif config['task'] == 'Direction_task' and saveTrail == '_angle':
                    offset += self.angleError(valY, prediction)
                elif config['task'] == 'Position_task':
                    offset += self.euclideanDistance(valY, prediction)
                else:
                    print("No corresponding error function")
                    return
            offset = offset / self.numberOfNetworks

            print("Evaluating PFI of {}".format(name))
            for k in range(iterations):
                for j in tqdm(range(int(dataShape[2]))):
                    valX = trainX.copy()
                    np.random.shuffle(valX[:, :, j])
                    for i in range(self.numberOfNetworks):
                        path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
                        trainer.ensemble.load_file_pattern = re.compile(saveTrail + name + '_nb_*', re.IGNORECASE)
                        trainer.load(path)
                        trainer.type = 'regressor'
                        prediction = np.squeeze(trainer.predict(valX))

                        if config['task'] == 'LR_task':
                            if useAccuracyBool:
                                prediction = np.rint(prediction)
                            electrodeLosses[j] += self.binaryCrossEntropyLoss(valY, prediction)
                        elif config['task'] == 'Direction_task' and saveTrail=='_amplitude':
                            electrodeLosses[j] += self.meanSquareError(valY, prediction)
                        elif config['task'] == 'Direction_task' and saveTrail=='_angle':
                            electrodeLosses[j] += self.angleError(valY, prediction)
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

        np.savetxt(config['model_dir'] +  'PFI' + nameSuffix + '.csv', results.transpose(), fmt='%s', delimiter=',', header=legend, comments='')
        return modelLosses

    def electrodeBarPlot(self,values,name="Electrode_Loss.png",format='pdf',colour='red'):
        values[np.where(values < 0)] = 0
        xAxis = np.arange(values.shape[0]) + 1
        fig = plt.figure()
        plt.xlabel("Electrode Number")
        plt.bar(xAxis, np.squeeze(values), color=colour)
        plt.legend()
        fig.savefig(config['model_dir'] + name + ".{}".format(format), format=format, transparent=True)
        plt.close()

    def electrodePlot(self, colourValues, filename="Electrode_Configuration", alpha=0.4, pathForOriginalRelativeToExecutable="./EEGEyeNet/Joels_Files/forPlot/"):
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
        cv2.imwrite(config['model_dir']+filename+'.png', img)

    def moveModels(self, newFolderName, originalPath, getEpochMetricsBool=True):
        """
        Moves all important files from an AnalEyeZor generated folder to another.
        @param newFolderName: Name of the new folder.
        @type newFolderName: String
        @param originalPath: Path within runs to the original Folder.
        @type originalPath: String
        @param getEpochMetricsBool: If True, generates Training Metrics from the console.out file.
        @type getEpochMetricsBool: Bool
        @return: None
        @rtype: None
        """
        try:
            os.mkdir(config['log_dir']+newFolderName)
        except:
            print("Folder already exists.")
            return
        if not os.path.isdir(originalPath+"checkpoint/"):
            print("Original Folder not found.")
            return

        #Move all networks
        runNames = os.listdir(originalPath+"checkpoint/")
        for runName in runNames:
            if not runName == ".DS_Store":
                networkNames = os.listdir(originalPath+"checkpoint/"+runName+"/")
                for networkName in networkNames:
                    for modelName in self.modelNames:
                        specialCaseBool = 'pyramidal' in networkName.lower()
                        if modelName.lower() in networkName.lower() and not specialCaseBool:
                            shutil.move(originalPath+"checkpoint/"+runName+"/"+networkName,config['log_dir']+newFolderName+"/checkpoint/"+runName+"/"+networkName)
                        #Special case since CNN and PyramidalCNN both contain the word CNN
                        elif modelName.lower() in networkName.lower() and specialCaseBool and 'pyramidal' in modelName.lower():
                            shutil.move(originalPath + "checkpoint/" + runName + "/" + networkName, config['log_dir'] + newFolderName + "/checkpoint/" + runName + "/" + networkName)
        #Get how many networks exists in this run.
        if os.path.exists(originalPath + "runs.csv"):
            allModelNames = pd.read_csv(originalPath+"runs.csv", usecols=["Model"])
        elif os.path.exists(originalPath + "runs_amplitude.csv"):
            allModelNames = pd.read_csv(originalPath + "runs_amplitude.csv", usecols=["Model"])
        elif os.path.exists(originalPath + "runs_angle.csv"):
            allModelNames = pd.read_csv(originalPath + "runs_angle.csv", usecols=["Model"])
        else:
            print("run not found")
            return

        #Generate Metrics.
        if os.path.exists(originalPath+"console.out") and getEpochMetricsBool:
            i = 1
            multiplier = 2 if config['task'] == "Direction_task" else 1
            with open(os.path.join(originalPath,"console.out")) as f:
                readingValuesBool = False
                metrics = None
                header = "Epoch"
                currentEpoch=0
                for line in f:
                    if i > len(allModelNames)*multiplier:
                        break
                    if "after" in line:
                        readingValuesBool = True
                    if "Epoch" in line:
                        currentEpoch = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", line)).astype(np.float)[0]
                    if "loss" in line and readingValuesBool:
                        metricToAppend = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", line)).astype(np.float)
                        metricToAppend = np.append(currentEpoch,metricToAppend[3:])
                        if metrics is None:
                            metrics = np.expand_dims(metricToAppend,0)
                            if metrics.shape[1] == 2:
                                header = 'Epoch,Loss'
                            elif metrics.shape[1] == 3:
                                header = 'Epoch,Loss,Accuracy'
                            elif metrics.shape[1] == 4:
                                header = 'Epoch,Loss,Accuracy,Val_Loss'
                            else:
                                header = 'Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy'
                        else:
                            metrics = np.concatenate((metrics, np.expand_dims(metricToAppend,0)), axis=0)
                    if "before" in line and readingValuesBool==True:
                        readingValuesBool = False
                        np.savetxt(config['log_dir']+newFolderName+"/"+str(allModelNames.values[i%len(allModelNames)-1][0])+'_{}.csv'.format(i), metrics, fmt='%s',
                                   delimiter=',', header=header, comments='')
                        i+=1
                        metrics = None
                if i <= len(allModelNames)*multiplier:
                    np.savetxt(config['log_dir'] + newFolderName + "/" + str(allModelNames.values[i%len(allModelNames)-1][0]) + '_{}.csv'.format(i),
                               metrics, fmt='%s',
                               delimiter=',', header=header, comments='')

        #Move important files.
        if os.path.exists(originalPath + "runs_angle.csv"):
            shutil.move(originalPath+"runs_angle.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "runs_amplitude.csv"):
            shutil.move(originalPath+"runs_amplitude.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "runs.csv"):
            shutil.move(originalPath+"runs.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "statistics_amplitude.csv"):
            shutil.move(originalPath + "statistics_amplitude.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "statistics.csv"):
            shutil.move(originalPath + "statistics.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "statistics_angle.csv"):
            shutil.move(originalPath + "statistics_angle.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "info.log"):
            shutil.move(originalPath + "info.log", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "config.csv"):
            shutil.move(originalPath + "config.csv", config['log_dir'] + newFolderName)
        if os.path.exists(originalPath + "console.out"):
            shutil.copy(originalPath + "console.out", config['log_dir'] + newFolderName)



        config['load_experiment_dir'] = newFolderName + "/"
        config['model_dir'] = config['log_dir'] + config['load_experiment_dir']
        config['checkpoint_dir'] = config['model_dir'] + 'checkpoint/'
        stamp = str(int(time.time()))
        config['info_log'] = config['model_dir'] + '/' + 'inference_info_' + stamp + '.log'
        config['batches_log'] = config['model_dir'] + '/' + 'inference_batches_' + stamp + '.log'

        self.currentFolderPath = config['log_dir'] + newFolderName + "/"

        if self.electrodes.shape[0] != 129:
            with open(config['model_dir'] + "electrodes" + '.txt', 'w') as f:
                f.write("Electrodes used:\n")
                f.write(self.electrodes)

    def colourCode(self, values, electrodes=np.arange(1,130), colourMap="Reds", minValue=5, epsilon = 0.01):
        #For Decibel, use epsilon = 1, for good colour visualisation use small epsilon > 0
        values[np.where(values < 0)] = 0
        values = 10*np.log(values + epsilon)
        cmap = cm.get_cmap(colourMap)
        norm = colors.Normalize(vmin=np.min(values[electrodes-1]), vmax=np.max(values[electrodes-1]))
        colours = cmap(norm(values))[:,0:3]
        colours[:,[2, 0]] = colours[:,[0, 2]]
        return colours * 255

    def plotTraining(self, modelFileName, filename='TrainingMetrics', columns=["Loss","Accuracy","Val_Loss","Val_Accuracy"], savePlotBool=True, format="pdf"):
        columns = ["Epoch"]+columns
        data = pd.read_csv(config['model_dir'] + modelFileName, usecols=columns).to_numpy()
        xAxis = np.arange(int(np.max(data[:,0])))+1

        #if columns[2]=="Accuracy":
        #    columns[2] = "Val_Loss"
        fig = plt.figure()
        plt.xlabel("Epoch")
        for i in range(1,data.shape[1]):
            plt.plot(xAxis, data[:,i], label=columns[i])
        plt.legend()

        if savePlotBool:
            fig.savefig(config['model_dir'] + filename+".{}".format(format), format=format, transparent=True)
        else:
            plt.show()
        plt.close()

    def topoPlot(self, values, filename='topoPlot', format='pdf',pathForOriginalRelativeToExecutable="./EEGEyeNet/Joels_Files/forPlot/", saveBool=True, cmap='Reds',epsilon=0.01):
        """

        @param values: Array of length 129, where the index + 1 equals the electrode number and a value, which will be colour coded. All values < 0 will be set to 0.
        @type values: Numpy Array
        @param filename: Name of the file as which the plot will be saved.
        @type filename: String
        @param format: Format of the save file.
        @type format: String
        @param pathForOriginalRelativeToExecutable: Depending from where the main.py script is called, this has to be adjusted to be the path relative to the starting file.
        @type pathForOriginalRelativeToExecutable: String
        @param saveBool: If True, the plot will be saved. Else it will be shown.
        @type saveBool: Bool
        @param cmap: Matplotlib colourmap
        @type cmap: String
        @param epsilon: Number to adjust weighting in the log plot. Has to be larger than 0.
        @type epsilon: float
        @return: None
        @rtype: None
        """

        if cmap not in plt.colormaps():
            print("Colourmap does not exist in Matplotlib.")
            cmap = "Reds"
        if epsilon <= 0:
            print("Epsilon too small, using epsilon = 1")
            epsilon = 1
        if values.shape[0] != 129:
            print("Wrong array dimensions")
            return

        electrodePositions = sio.loadmat(pathForOriginalRelativeToExecutable+"lay129_head.mat")['lay129_head']['pos'][0][0]
        outline = sio.loadmat(pathForOriginalRelativeToExecutable+"lay129_head.mat")['lay129_head']['outline'][0][0]
        mask = sio.loadmat(pathForOriginalRelativeToExecutable+"lay129_head.mat")['lay129_head']['mask'][0][0]
        values[np.where(values < 0)] = 0
        values = 10 * np.log(values + epsilon)
        fig = plt.figure(figsize=(7,4.5))
        #Generating outline dictionary for mne topoplot
        outlines = dict()
        outlines["mask_pos"] = (mask[0,0][:,0],mask[0,0][:,1])
        outlines["head"] = (outline[0, 0][:,0],outline[0, 0][:,1])
        outlines["nose"] = (outline[0, 1][:,0],outline[0, 1][:,1])
        outlines["ear_left"] = (outline[0, 2][:,0],outline[0, 2][:,1])
        outlines["ear_right"] = (outline[0, 3][:,0],outline[0, 3][:,1])
        #This cuts out parts of the colour circle
        outlines['clip_radius'] = (0.5,) * 2
        outlines['clip_origin'] = (0,0.07)
        im, cm = mne.viz.plot_topomap(np.squeeze(values),electrodePositions[3:132,:],outlines=outlines,show=False,cmap=cmap)
        clb = fig.colorbar(im)
        if epsilon==1:
            clb.ax.set_title("Loss in Db")
        else:
            clb.ax.set_title("10x Log-Ratio-Loss, eps={}".format(epsilon))
        plt.legend()
        if saveBool:
            fig.savefig(config['model_dir'] + filename + ".{}".format(format), format=format, transparent=True)
        else:
            plt.show()
        plt.close()

    def visualizePrediction(self, modelName, run = 1, nrOfPoints=9, filename='predictionVisualisation', format='pdf',saveBool=True, scale=False):

        trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1]
        ids = trainY[:, 0]
        trainIndices, valIndices, testIndices = split(ids, 0.7, 0.15, 0.15)
        if scale:
            trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:, :, self.electrodes.astype(np.int) - 1]
            logging.info('Standard Scaling')
            scaler = StandardScaler()
            scaler.fit(trainX[trainIndices])
            trainX = scaler.transform(trainX[valIndices])
        else:
            trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][testIndices, :,:]
            trainX = trainX[:, :,self.electrodes.astype(np.int) - 1]
        valY = trainY[testIndices,1:]
        trainX, valY  = trainX[:nrOfPoints], valY[:nrOfPoints,:]
        del trainIndices, valIndices, testIndices, trainY
        if config['task'] == 'Direction_task':
            modelAmp = all_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][modelName]
            modelAngle = all_models[config['task']][config['dataset']][config['preprocessing']]['angle'][modelName]

            path = config['checkpoint_dir'] + 'run' + str(run) + '/'

            trainerAmp = modelAmp[0](**modelAmp[1])
            trainerAmp.ensemble.load_file_pattern = re.compile('_amplitude' + modelName + '_nb_*', re.IGNORECASE)
            trainerAmp.load(path)
            predictionAmp = np.squeeze(trainerAmp.predict(trainX))

            trainerAngle = modelAngle[0](**modelAngle[1])
            trainerAngle.ensemble.load_file_pattern = re.compile('_angle' + modelName + '_nb_*', re.IGNORECASE)
            trainerAngle.load(path)
            predictionAngle = np.squeeze(trainerAngle.predict(trainX))

            prediction = np.zeros([predictionAngle.shape[0],2])
            prediction[:,0] = np.multiply(predictionAmp, np.cos(predictionAngle))
            prediction[:,1] = np.multiply(predictionAmp, np.sin(predictionAngle))

            truth = np.zeros([predictionAngle.shape[0], 2])
            truth[:,0] = np.multiply(valY[:,0],np.cos(valY[:,1]))
            truth[:, 1] = np.multiply(valY[:,0], np.sin(valY[:,1]))
        elif config['task'] == 'Position_task':
            model = all_models[config['task']][config['dataset']][config['preprocessing']][modelName]
            trainer = model[0](**model[1])
            trainer.ensemble.load_file_pattern = re.compile(modelName + '_nb_*', re.IGNORECASE)
            path = config['checkpoint_dir'] + 'run' + str(run) + '/'
            trainer.load(path)
            prediction = np.squeeze(trainer.predict(trainX))
            truth = valY
        else:
            print("Task not yet configured.")
            return
        steps = np.linspace(0.3,1,num=int(nrOfPoints/3)+1)
        colour = np.zeros((nrOfPoints,3))
        for i in range(3):
            length = colour[int(i*nrOfPoints/3):int((1+i)*nrOfPoints/3),i].shape[0]
            colour[int(i*nrOfPoints/3):int((1+i)*nrOfPoints/3),i] =  steps[:length]

        fig = plt.figure()
        plt.scatter(truth[:,0],truth[:,1],c=colour, marker='o',label="Ground Truth")
        plt.scatter(prediction[:, 0], prediction[:, 1], c=colour, marker='x',label="Prediction")

        for i in range(prediction.shape[0]):
            plt.plot(np.array([prediction[i,0],truth[i,0]]),np.array([prediction[i,1],truth[i,1]]),c=colour[i])

        plt.axhline(0, color='black',linewidth=0.1)
        plt.axvline(0, color='black',linewidth=0.1)

        plt.legend()
        if saveBool:
            fig.savefig(config['model_dir'] + filename + ".{}".format(format), format=format, transparent=True)
        else:
            plt.show()
        plt.close()

    def generateTable(self,modelFileName,filename='tableLatex',addNrOfParams=True,caption="Performance of each Network"):

        data = pd.read_csv(config['model_dir'] + modelFileName,header=None)
        data = data.astype(str).values.tolist()
        if addNrOfParams:
            data[0] = data[0] + ["\#Parameters"]
            for i in range(1,len(data)):
                modelName=data[i][0]
                if 'amplitude' in modelFileName.lower():
                    model = all_models[config['task']][config['dataset']][config['preprocessing']]['amplitude'][modelName]
                    saveTrail='_amplitude'
                elif 'angle' in modelFileName.lower():
                    model = all_models[config['task']][config['dataset']][config['preprocessing']]['angle'][modelName]
                    saveTrail = '_angle'
                else:
                    model = all_models[config['task']][config['dataset']][config['preprocessing']][modelName]
                    saveTrail = ''

                path = config['checkpoint_dir'] + 'run' + str(1) + '/'

                trainer = model[0](**model[1])
                trainer.ensemble.load_file_pattern = re.compile(saveTrail + modelName + '_nb_*', re.IGNORECASE)
                trainer.load(path)


                trainableParams = np.sum([np.prod(v.get_shape()) for v in trainer.ensemble.models[0].trainable_weights])
                nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in trainer.ensemble.models[0].non_trainable_weights])
                totalParams = trainableParams + nonTrainableParams
                data[i] = data[i] + [str(totalParams*trainer.ensemble.nb_models)]

        table = Texttable()
        table.set_cols_align(["c"] * len(data[0]))
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(data)
        with open(config['model_dir']+filename+'.txt', 'w') as f:
            f.write(latextable.draw_latex(table, caption=caption))

    def combineResults(self,modelFileName,directories,filename,columns=["Model","Mean_score","Std_score","Mean_runtime","Std_runtime"],nameColumn="Model",nameStartIndex=0):
        config['log_dir']
        data = pd.read_csv(config['model_dir'] + directories[0] + modelFileName, header=None, usecols=columns)
        data[nameColumn] = directories[0][nameStartIndex:] + data[nameColumn].astype(str)

        for i in range(1,len(directories)):
            toAppend = pd.read_csv(config['model_dir'] + directories[i] + modelFileName, usecols=columns)
            toAppend[nameColumn] = directories[i][nameStartIndex:] + "_" + toAppend[nameColumn].astype(str)
            data.append(toAppend)

        data.to_csv(config['model_dir'] + filename, sep='\t')

    def meanSquareError(self,y,yPred):
        return np.sqrt(mean_squared_error(y, yPred.ravel()))

    def angleError(self,y,yPred):
        return np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - yPred.ravel()), np.cos(y - yPred.ravel())))))

    def euclideanDistance(self,y,yPred):
        return np.linalg.norm(y - yPred, axis=1).mean()

    def binaryCrossEntropyLoss(self,y,yPred):
        return log_loss(y,yPred, normalize=True)



