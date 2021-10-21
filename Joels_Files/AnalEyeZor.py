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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf
import shutil
import pandas as pd
import re

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

    def __init__(self, task, dataset, preprocessing, models, featureExtraction = False, trainBool = True, saveModelBool = True, path=None):

        config['include_ML_models'] = False
        config['include_DL_models'] = False
        config['include_your_models'] = True
        config['include_dummy_models'] = False
        config['feature_extraction'] = featureExtraction

        self.inputShape = (1, 258) if config['feature_extraction'] else (500, 129)
        self.numberOfVotingNetworks = 5
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

        #########################################Loading_or_Training_Model####################################################
        your_models[config['task']] = {config['dataset']:{config['preprocessing']:{}}}
        for i in models:

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
            self.currentFolderPath = config['model_dir']
            logging.info("------------------------------Loading the Data------------------------------")
            trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
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



    def PFI(self, scale = False):
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
        electrodeLosses = np.zeros(dataShape[2])
        models = all_models[config['task']][config['dataset']][config['preprocessing']]
        for name, model in models.items():
            start_time = time.time()
            for i in range(self.numberOfVotingNetworks):
                path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
                trainer = tf.keras.models.load_model(path+name)
                print("Evaluating {}, run {}".format(name,i+1))
                for j in tqdm(range(int(dataShape[2]))):

                    valX = trainX.copy()
                    np.random.shuffle(valX[:,:,j])

                    if config['task'] == 'LR_task':
                        electrodeLosses[j] += self.accuracyLoss(valY, trainer.predict(valX))
                        print(electrodeLosses[j])
                    elif config['task'] == 'Direction_task':
                        electrodeLosses[j] += self.meanSquareError(valY, trainer.predict(valX))
                    elif config['task'] == 'Position_task':
                        electrodeLosses[j] += self.euclideanDistance(valY, trainer.predict(valX))
            modelLosses[name] = electrodeLosses / self.numberOfVotingNetworks
            runtime = (time.time() - start_time)
            logging.info("--- Sorted Electrodes According to Influence for {}:".format(name))
            logging.info(1+(np.argsort(modelLosses[name]))[::-1])
            logging.info("--- Losses of each Electrode for {}:".format(name))
            logging.info(modelLosses[name][((np.argsort(modelLosses[name]))[::-1])])
            logging.info("--- Runtime: %s for seconds ---" % runtime)
        logging.info("------------------------------Evaluated Electrodes------------------------------")
        return modelLosses

    #def electrodePlot(self, values):

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
                    if modelName.lower() in networkName.lower():
                        shutil.move(originalPath+"checkpoint/"+runName+"/"+networkName,config['log_dir']+newFolderName+"/checkpoint/"+runName)

        allModelNames = pd.read_csv(originalPath+"runs.csv", usecols=["Model"])
        if os.path.exists(originalPath+"console.out") and getEpochMetricsBool:
            i = 1
            with open(os.path.join(originalPath,"console.out")) as f:
                readingValuesBool = False
                metrics = np.zeros([1,4])
                for line in f:
                    if i > len(allModelNames):
                        break
                    if "after" in line:
                        readingValuesBool = True
                    if "loss" in line and readingValuesBool:
                        metricToAppend = np.array(re.findall(r"[-+]?\d*\.\d+|\d+", line)).astype(np.float)
                        metrics = np.concatenate((metrics, np.expand_dims(metricToAppend[3:],0)), axis=0)
                    if "before" in line and readingValuesBool==True:
                        readingValuesBool = False
                        np.savetxt(config['log_dir']+newFolderName+"/"+str(allModelNames.values[i-1][0])+'_{}.csv'.format(i), metrics[1:,:], fmt='%s',
                                   delimiter=',', header='Loss,Accuracy,Val_Loss,Val_Accuracy', comments='')
                        i+=1
                        metrics = np.zeros([1, 4])
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

    def meanSquareError(self,y,yPred):
        return np.sqrt(mean_squared_error(y, yPred.ravel()))

    def angleError(self,y,yPred):
        np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - yPred.ravel()), np.cos(y - yPred.ravel())))))

    def euclideanDistance(self,y,yPred):
        return np.linalg.norm(y - yPred, axis=1).mean()

    def accuracyLoss(self,y,yPred):
        return 1-np.mean(y==np.rint(np.squeeze(yPred)).astype(int))












    def __trainModel(self,N):

        def split(ids, train, val, test):
            assert (train + val + test == 1)
            IDs = np.unique(ids)
            num_ids = len(IDs)

            # priority given to the test/val sets
            test_split = math.ceil(test * num_ids)
            val_split = math.ceil(val * num_ids)
            train_split = num_ids - val_split - test_split

            train = np.isin(ids, IDs[:train_split])
            val = np.isin(ids, IDs[train_split:train_split + val_split])
            test = np.isin(ids, IDs[train_split + val_split:])

            return train, val, test

        def try_models(trainX, trainY, ids, models, iteration, N=5, scoring=None, scale=False, save_trail='', save=False):

            logging.info("Training the models")
            train, val, test = split(ids, 0.7, 0.15, 0.15)
            X_train, y_train = trainX[train], trainY[train]
            X_val, y_val = trainX[val], trainY[val]
            X_test, y_test = trainX[test], trainY[test]

            if scale:
                logging.info('Standard Scaling')
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

            all_runs = []
            statistics = []

            for name, model in models.items():
                logging.info("Training of " + name)

                model_runs = []

                for i in range(N):
                    # create the model with the corresponding parameters
                    trainer = model[0](**model[1])
                    logging.info(trainer)
                    start_time = time.time()

                    # Taking care of saving and loading
                    path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    if config['retrain'] and iteration == 0:
                        trainer.fit(X_train, y_train, X_val, y_val)
                    elif config['retrain'] and iteration > 0:
                        trainer.load(path)
                        trainer.fit(X_train, y_train, X_val, y_val)
                    else:
                        trainer.load(path)

                    if config['save_models']:
                        trainer.save(path)

                    # pred = trainer.predict(X_test)
                    # print(y_test.shape)
                    # print(pred.shape)
                    score = scoring(y_test, trainer.predict(X_test))
                    # print(score)

                    runtime = (time.time() - start_time)
                    all_runs.append([name, score, runtime])
                    model_runs.append([score, runtime])

                    logging.info("--- Score: %s " % score)
                    logging.info("--- Runtime: %s for seconds ---" % runtime)

                model_runs = np.array(model_runs)
                model_scores, model_runtimes = model_runs[:, 0], model_runs[:, 1]
                statistics.append(
                    [name, model_scores.mean(), model_scores.std(), model_runtimes.mean(), model_runtimes.std()])

            np.savetxt(config['model_dir'] + '/runs' + save_trail + '.csv', all_runs, fmt='%s', delimiter=',',
                       header='Model,Score,Runtime', comments='')
            np.savetxt(config['model_dir'] + '/statistics' + save_trail + '.csv', statistics, fmt='%s', delimiter=',',
                       header='Model,Mean_score,Std_score,Mean_runtime,Std_runtime', comments='')

        def benchmark(trainX, trainY, iteration):
            np.savetxt(config['model_dir'] + '/config.csv',
                       [config['task'], config['dataset'], config['preprocessing']], fmt='%s')
            models = all_models[config['task']][config['dataset']][config['preprocessing']]

            ids = trainY[:, 0]

            if config['task'] == 'LR_task':
                if config['dataset'] == 'antisaccade':
                    scoring = (lambda y, y_pred: accuracy_score(y,
                                                                y_pred.ravel()))  # Subject to change to mean euclidean distance.
                    y = trainY[:, 1]  # The first column are the Id-s, we take the second which are labels
                    try_models(trainX=trainX, trainY=y, ids=ids, models=models, iteration=iteration, scoring=scoring)
                else:
                    raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

            elif config['task'] == 'Direction_task':
                if config['dataset'] == 'dots':
                    scoring = (lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred.ravel())))
                    y1 = trainY[:, 1]  # The first column are the Id-s, we take the second which are amplitude labels
                    try_models(trainX=trainX, trainY=y1, ids=ids, models=models['amplitude'], iteration=iteration, scoring=scoring,
                               save_trail='_amplitude')
                    scoring2 = (lambda y, y_pred: np.sqrt(
                        np.mean(np.square(np.arctan2(np.sin(y - y_pred.ravel()), np.cos(y - y_pred.ravel()))))))
                    y2 = trainY[:,
                         2]  # The first column are the Id-s, second are the amplitude labels, we take the third which are the angle labels
                    try_models(trainX=trainX, trainY=y2, ids=ids, models=models['angle'], iteration=iteration, scoring=scoring2,
                               save_trail='_angle')
                else:
                    raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

            elif config['task'] == 'Position_task':
                if config['dataset'] == 'dots':
                    scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean())  # Euclidean distance
                    y = trainY[:,
                        1:]  # The first column are the Id-s, the second and third are position x and y which we use
                    try_models(trainX=trainX, trainY=y, ids=ids, models=models, iteration=iteration, scoring=scoring2)
                else:
                    raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")
            else:
                raise NotImplementedError(f"Task {config['task']} is not implemented yet.")

        dataSize = np.shape(IOHelper.get_npz_data(config['data_dir'], verbose=True)[0])[0]
        for i in range(N):
            trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][int(i/N*dataSize):int(i+1/N*dataSize)]
            trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1][int(i/N*dataSize):int(i+1/N*dataSize)]
            benchmark(trainX, trainY, i)