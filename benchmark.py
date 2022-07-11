import numpy as np
import logging
import time
import math
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from config import config, create_folder
from hyperparameters import all_models
import os
import sys
from utils import IOHelper
import wandb

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

# return boolean arrays with length corresponding to n_samples
# the split is done based on the number of IDs
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

    return train, val, test


def try_models(trainX, trainY, ids, models, N=1, scoring=None, scale=True, save_trail='', save=False):

    logging.info("Training the models")
    train, val, test = split(ids, 0.7, 0.15, 0.15)
    X_train, y_train = trainX[train], trainY[train]
    X_val, y_val = trainX[val], trainY[val]
    X_test, y_test = trainX[test], trainY[test]

    if scale:
        logging.info('Standard Scaling')
        scaler = StandardScaler()
        # Reshape, fit on training data, shape back 
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))
        scaler.fit_transform(X_train)
        X_train = X_train.reshape((nsamples, nx, ny))
        # Reshape, transform, shape back on validation 
        nsamples, nx, ny = X_val.shape
        X_val = X_val.reshape((nsamples,nx*ny))
        X_val = scaler.transform(X_val)
        X_val = X_val.reshape((nsamples, nx, ny))
        # Reshape, transform, shape back on test 
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))
        X_test = scaler.transform(X_test)
        X_test = X_test.reshape((nsamples, nx, ny))
    
    all_runs = []
    statistics = []

    for name, model in models.items():
        logging.info("------------------------------------------------------------------------------------")
        logging.info("Training of " + name)

        model_runs = []

        for i in range(N):
            # create the model with the corresponding parameters

            # Taking care of saving and loading
            path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
            if not os.path.exists(path):
                os.makedirs(path)

            trainer = model[0](path=path, **model[1])
            start_time = time.time()

            # W&B Init
            run = wandb.init(project=config['project'], entity=config['entity'])
            wandb.run.name = config['task'] + '_' + name + "_model_nb_{}_".format(str(N)) + wandb.run.name
            wandb.config.update({**model[1]})

            if config['retrain']:
                trainer.fit(X_train, y_train, X_val, y_val)
            else:
                trainer.load(path)

            if config['save_models'] and config['include_ML_models']: # DL models save only based on validation set metric 
                trainer.save(path)

            if config['include_DL_models']:
                logging.info(f"Loading best validation model to compute score on test set...")
                trainer.load(path) # load the best model on validation loss

            logging.info(f"Scoring...")
            pred = trainer.predict(X_test)
            #print("gt", y_test.shape)
            #print("pred", pred.shape)
            #pred_max = np.argmax(np.reshape(pred, (-1,3,500)), axis=1)
            #print("pred max", pred_max.shape)
            #print("y", y_test[:25])
            #print("pred_max", pred_max[:25])
            min_len = min(len(y_test), len(pred)) # we might drop the last batch 
            score = scoring(y_test[:min_len], pred[:min_len])
            #print(score)

            runtime = (time.time() - start_time)
            all_runs.append([name, score, runtime])
            model_runs.append([score, runtime])

            logging.info("--- Score: %s " % score)
            logging.info("--- Runtime: %s for seconds ---" % runtime)

            run.finish()
        
        model_runs = np.array(model_runs)
        model_scores, model_runtimes = model_runs[:,0], model_runs[:,1]
        statistics.append([name, model_scores.mean(), model_scores.std(), model_runtimes.mean(), model_runtimes.std()])

    np.savetxt(config['model_dir']+'/runs'+save_trail+'.csv', all_runs, fmt='%s', delimiter=',', header='Model,Score,Runtime', comments='')
    np.savetxt(config['model_dir']+'/statistics'+save_trail+'.csv', statistics, fmt='%s', delimiter=',', header='Model,Mean_score,Std_score,Mean_runtime,Std_runtime', comments='')
    logging.info("-----------------")
    logging.info("-----------------")
    logging.info("-----------------")
    logging.info("Finished benchmark.")

def benchmark():

    # Setting up logging
    create_folder()
    logging.basicConfig(filename=config['info_log'], level=logging.INFO)
    logging.info('Started the Logging')
    logging.info(f"Using {config['framework']}")
    start_time = time.time()

    # For being able to see progress that some methods use verbose (for debugging purposes)
    f = open(config['model_dir'] + '/console.out', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    #Load the data
    electrodeIndices = np.array(config['electrodes']).astype(np.int)
    trainX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:, :, electrodeIndices-1]
    trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1]

    np.savetxt(config['model_dir']+'/config.csv', [config['task'], config['dataset'], config['preprocessing'], np.array2string(electrodeIndices)], fmt='%s')
    models = all_models[config['task']][config['dataset']][config['preprocessing']]

    ids = trainY[:, 0]

    if config['task'] == 'LR_task':
        if config['dataset'] == 'antisaccade':
            scoring = (lambda y, y_pred: accuracy_score(y, y_pred.ravel()))  # Subject to change to mean euclidean distance.
            y = trainY[:,1] # The first column are the Id-s, we take the second which are labels
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Direction_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred.ravel())))
            y1 = trainY[:,1] # The first column are the Id-s, we take the second which are amplitude labels
            try_models(trainX=trainX, trainY=y1, ids=ids, models=models['amplitude'], scoring=scoring, save_trail='_amplitude')
            scoring2 = (lambda y, y_pred: np.sqrt(np.mean(np.square(np.arctan2(np.sin(y - y_pred.ravel()), np.cos(y - y_pred.ravel()))))))
            y2 = trainY[:,2] # The first column are the Id-s, second are the amplitude labels, we take the third which are the angle labels
            try_models(trainX=trainX, trainY=y2, ids=ids, models=models['angle'], scoring=scoring2, save_trail='_angle')
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Position_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred))) # Subject to change to mean euclidean distance.
            scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean()) # Euclidean distance
            y = trainY[:,1:] # The first column are the Id-s, the second and third are position x and y which we use
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring2)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Segmentation_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : f1_score(np.concatenate(y), np.concatenate(np.argmax(np.reshape(np.concatenate(y_pred), (-1,3,500)), axis=1)), average='macro')) # Macro average f1 as segmentation metric 
            y = trainY[:,1:] # The first column are the Id-s, the rest are the labels of the events (0=F, 1=S, 2=B)
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Path_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred))) # Subject to change to mean euclidean distance.
            scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean()) # Euclidean distance
            y = trainY[:,1:] # The first column are the Id-s, the second and third are position x and y which we use
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring2)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented yet.")

    logging.info("--- Runtime: %s seconds ---" % (time.time() - start_time))
    logging.info('Finished Logging')