from config import config
import logging
import torch
import os
import numpy as np
import re

from DL_Models.torch_models.torch_utils.dataloader import create_dataloader
from DL_Models.torch_models.torch_utils.training import test_loop
from DL_Models.torch_models.torch_utils.plot import plot_metrics


class Ensemble_torch:
    """
    The Ensemble is a model itself, which contains a number of models whose prediction is averaged (majority decision in case of a classifier). 
    """

    def __init__(self, path, model_name='CNN', nb_models=5, loss='bce', batch_size=64, **model_params):
        """
        model_name: the model that the ensemble uses
        nb_models: Number of models to run in the ensemble
        model_list: optional, give a list of models that should be contained in the Ensemble
        ...
        """
        self.model_name = model_name
        self.nb_models = nb_models
        self.model_params = model_params
        self.batch_size = batch_size
        self.loss = loss
        self.path = path 
        self.model_instance = None
        self.load_file_pattern = re.compile(self.model_name + f'_nb_..', re.IGNORECASE)
        self.models = []

        if self.model_name == 'CNN':
            from DL_Models.torch_models.CNN.CNN import CNN
            self.model = CNN
        elif self.model_name == 'EEGNet':
            from DL_Models.torch_models.EEGNet.eegNet import EEGNet
            self.model = EEGNet
        elif self.model_name == 'InceptionTime':
            from DL_Models.torch_models.InceptionTime.InceptionTime import Inception
            self.model = Inception
        elif self.model_name == 'PyramidalCNN':
            from DL_Models.torch_models.PyramidalCNN.PyramidalCNN import PyramidalCNN
            self.model = PyramidalCNN
        elif self.model_name == 'Xception':
            from DL_Models.torch_models.Xception.Xception import XCEPTION
            self.model = XCEPTION
        elif self.model_name == 'biLSTM':
            from DL_Models.torch_models.BiLSTM.biLSTM import biLSTM
            self.model = biLSTM
        elif self.model_name == 'GCN':
            from DL_Models.torch_models.GCN.GCN import GCN
            self.model = GCN
        elif self.model_name == 'LSTM':
            from DL_Models.torch_models.LSTM.LSTM import LSTM
            self.model = LSTM 
        elif self.model_name == 'UNet':
            from DL_Models.torch_models.UNet.UNet import UNet
            self.model = UNet
        elif self.model_name == 'TransformerSimple':
            from DL_Models.torch_models.Transformer.TransformerSimple import TransformerSimple 
            self.model = TransformerSimple
        elif self.model_name == 'ConvLSTM':
            from DL_Models.torch_models.ConvLSTM.ConvLSTM import ConvLSTM
            self.model = ConvLSTM
        else:
            raise Exception("choose valid model")

    
    def fit(self, trainX, trainY, validX, validY):
        """
        Fit an ensemble of models. They will be saved by BaseNet into the model dir
        """

        # Create dataloaders
        trainX = np.transpose(trainX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        validX = np.transpose(validX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        train_dataloader = create_dataloader(trainX, trainY, self.batch_size, self.model_name)
        validation_dataloader = create_dataloader(validX, validY, self.batch_size, self.model_name)
        # Fit the models 
        for i in range(self.nb_models):
            logging.info(f'Start fitting ensemble model {self.model_name} {i}/{self.nb_models-1} ...')
            model = self.model(path=self.path, model_name=self.model_name, loss = self.loss, model_number=i, batch_size=self.batch_size, **self.model_params)
            train_loss, val_loss = model.fit(train_dataloader, validation_dataloader)
            plot_metrics(train=train_loss, val=val_loss, model_name=self.model_name, metric=self.loss, output_dir=config['checkpoint_dir'] + 'run' + str(i + 1) + '/')
            self.models.append(model)
            logging.info('Finished fitting ensemble model {}/{} ...'.format(i, self.nb_models-1))


    def predict(self, testX):
        # remove this hack 
        testX = np.transpose(testX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        #a,b,c = testX.shape
        #a = self.batch_size - a % self.batch_size
        #dummy = np.zeros((a,b,c))
        #testX = np.concatenate((testX, dummy)) # TO ADD batch_size - testX.shape[0]%batch_size
        test_dataloader = create_dataloader(testX, testX, self.batch_size, self.model_name, drop_last=True)
        pred = None
        #print(f"self models len {len(self.models)}")
        for model in self.models:
            if torch.cuda.is_available():
                model.cuda()
            if pred is not None:
                pred += test_loop(dataloader=test_dataloader, model=model)
            else:
                pred = test_loop(dataloader=test_dataloader, model=model)
        #pred = pred[:-a]
        return pred / len(self.models) 

    def save(self, path):
        for i, model in enumerate(self.models):
            ckpt_dir = path + self.model_name + '_nb_{}_'.format(i) + '.pth'
            torch.save(model.state_dict(), ckpt_dir)

    def load(self, path):
        #print(f"cuda avail {torch.cuda.is_available()}")
        self.models = []
        i = 0
        logging.info(f"Loading model type {self.model_name}")
        for file in os.listdir(path):
            if not self.load_file_pattern.match(file):
                continue
            logging.info(f"Loading model {i} from file {file} and predict with it")
            i += 1
            model = self.model(path=self.path, model_name=self.model_name, loss=self.loss, model_number=-1, batch_size=self.batch_size,
                               **self.model_params) 
            #print(path + file)
            model.load_state_dict(torch.load(path + file))  # model.load_state_dict(torch.load(PATH))
            model.eval()  # needed before prediction
            self.models.append(model)