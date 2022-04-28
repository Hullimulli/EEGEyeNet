import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import logging
from config import config
import wandb
from tqdm import tqdm
from Joels_Files.plotFunctions.prediction_visualisations import getVisualisation
import matplotlib.pyplot as plt
import sys

class prediction_history(tf.keras.callbacks.Callback):
    """
    Prediction history for model ensembles=
    """
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.predhis = []
        self.targets = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        self.predhis.append(y_pred)


class BaseNet:
    def __init__(self, loss, input_shape, output_shape, epochs=50, verbose=True, model_number=0):
        self.epochs = epochs
        self.verbose = verbose
        self.model_number = model_number
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.early_stopped = False
        self.loss = loss 
        self.nb_channels = input_shape[1]
        self.timesamples = input_shape[0]
        self.model = self._build_model()

        # Compile the model depending on the task 
        if self.loss == 'bce':
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), metrics=['accuracy'])
        elif self.loss == 'mse':
            self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), metrics=['mean_squared_error'])
        elif self.loss == 'angle-loss':
            from DL_Models.tf_models.utils.losses import angle_loss
            self.model.compile(loss=angle_loss, optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']))
        else:
            raise ValueError("Choose valid loss for your task")
            
        # if self.verbose:
            # self.model.summary()

        logging.info(f"Number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])}")
        logging.info(f"Number of non-trainable parameters: {np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])}")

    # abstract method
    def _split_model(self):
        pass

    # abstract method
    def _build_model(self):
        pass

    def get_model(self):
        return self.model

    def save(self, path):
        self.model.save(path)

    def fit(self, X_train, y_train, X_val, y_val):
        #csv_logger = CSVLogger(config['batches_log'], append=True, separator=';')
        #ckpt_dir = config['model_dir'] + '/best_models/' + config['model'] + '_nb_{}_'.format(self.model_number) + 'best_model.h5'
        #ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_dir, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

        # W&B Init
        run = wandb.init(project=config['project'], entity=config['entity'])
        wandb.run.name = config['task'] + '_' + str(self) + "_model_nb_{}_".format(str(self.model_number)) + wandb.run.name
        wandb.config = {
            "model_name": str(self) + '_run_' + str(self.model_number),
            "learning_rate": config['learning_rate'],
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "task": config['task']
        }

        best_val_loss = sys.maxsize  # For early stopping
        val_loss_epoch = sys.maxsize
        patience = 0
        for i in tqdm(range(self.epochs)):
            logging.info("-------------------------------")
            logging.info(f"Epoch {i+1}")
            if not self.early_stopped:
                hist = self.model.fit(X_train, y_train, verbose=2, batch_size=self.batch_size, validation_data=(X_val, y_val),
                                        epochs=i+1, initial_epoch=i)
                #W&B Logs
                logs = {str(key): value[0] for key, value in hist.history.items()}
                val_loss_epoch = logs['val_loss']
                prediction = np.squeeze(self.model.predict(X_val))
                if self.loss == "angle-loss":
                    addLogs = { "visualisation": wandb.Image(getVisualisation(groundTruth=y_val,
                                                         prediction=np.expand_dims(prediction,axis=(0,1)),
                                                         modelName="Model",anglePartBool=True)),
                                "epoch": i+1}
                else:
                    addLogs = { "visualisation": wandb.Image(getVisualisation(groundTruth=y_val,
                                                         prediction=np.expand_dims(prediction,axis=(0,1)),
                                                         modelName="Model",anglePartBool=False)),
                                "epoch": i+1}
                wandb.log({**logs,**addLogs})
                plt.close('all')

            # Impementation of early stopping
            if config['early_stopping'] and not self.early_stopped:
                if patience > config['patience']:
                    logging.info(f"Early stopping the model after {i} epochs")
                    self.early_stopped = True
                if val_loss_epoch >= best_val_loss:
                    logging.info(f"Validation loss did not improve, best was {best_val_loss}")
                    patience +=1
                else:
                    best_val_loss = val_loss_epoch
                    logging.info(f"Improved validation loss to: {best_val_loss}")
                    patience = 0
        run.finish()

    def predict(self, testX):
        return self.model.predict(testX)
