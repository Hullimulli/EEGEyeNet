from .dataLoader import loadData, split
from .resCNN import resCNN
import os
import tensorflow.keras as keras
import wandb
from tqdm import tqdm
import numpy as np
import time
import tensorflow as tf
from .updateModel import mseUpdate,angleLossUpdate, mse, angle_loss
from config import config
from .preprocess import convertToImage

class method:

    def __init__(self,name:str = 'resCNN',directory: str = "./", imageShape:(int,int)=(32,32), nrOfSamples:int = 500,
                 batchSize: int = 32, nrOfEpochs:int = 10,wandbProject:str = "", continueTrainingBool: bool = False,
                 loss:str = 'mse'):
        self.model = None
        self.name = name
        self.epochs = nrOfEpochs
        self.batchSize = batchSize
        self.nrOfSamples = nrOfSamples
        self.checkpointPath = os.path.join(directory, self.name)
        self.inputShape = (imageShape[0], imageShape[1],self.nrOfSamples)
        self.architecture = resCNN()
        self.learningRate = 0.0001
        self.wandbProject = wandbProject


        if loss=='angle-loss':
            self.update = angleLossUpdate
            self.loss = angle_loss
        else:
            self.update = mseUpdate
            self.loss = mse

        self.model = self.architecture.buildModel(inputShape=self.inputShape)
        if not os.path.exists(self.checkpointPath):
            os.mkdir(self.checkpointPath)
        elif continueTrainingBool:
            try:
                self.model = keras.models.load_model(self.checkpointPath)
            except:
                pass


    def fit(self):
        if self.wandbProject != "":
            run = wandb.init(project=self.wandbProject, entity='hullimulli')
            wandb.run.name = self.name + "_" + wandb.run.name
            wandbConfig = wandb.config
        print(self.model.summary())
        inputPath = config['data_dir'] + config['all_EEG_file'][:-4] + '_X.npy'
        targetPath = config['data_dir'] + config['all_EEG_file'][:-4] + '_Y.npy'
        targetIndex = 1
        inputs, targets = loadData(inputPath,targetPath)
        trainIndices, valIndices, testIndices = split(targets[:,0], 0.7, 0.15, 0.15)
        if self.model is None:
            self.model = self.architecture.buildModel(inputShape=self.inputShape)

        trainable_count = int(
            np.sum([keras.backend.count_params(p) for p in self.model.trainable_weights]))
        non_trainable_count = int(
            np.sum([keras.backend.count_params(p) for p in self.model.non_trainable_weights]))

        if self.wandbProject != "":
            wandbConfig.update({"Directory": self.checkpointPath, "Learning_Rate": self.learningRate, "Nr_Samples": self.nrOfSamples,
                           "Patch_Shape": "{},{}".format(self.inputShape[1], self.inputShape[2]), "Training Set": inputPath})

        pbar = tqdm(range(self.epochs))
        valLoss = np.inf
        trainingTime = time.time()
        for e in pbar:
            loss_values = 0
            tic = time.time()

            # Permute
            p = np.random.permutation(len(trainIndices))

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
            nrOfBatches = 0.0
            totNrOfBatches = len(p) / self.batchSize
            for batch in range(0, len(p), self.batchSize):
                input = convertToImage(inputs[p[batch:batch + self.batchSize]])
                loss_values_temp, grads = self.update(model=self.model, input=input,
                                                    ground=targets[p[batch:batch + self.batchSize],targetIndex])
                loss_values += loss_values_temp
                if batch + self.batchSize > len(p):
                    nrOfBatches += (len(p) - batch) / self.batchSize
                else:
                    nrOfBatches += 1
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                estimateOfArrival = int((time.time() - tic) / nrOfBatches * (totNrOfBatches-nrOfBatches)/60)
                pbar.set_description(
                    "epoch: {}, patch {}/{}, loss: {}, eta: {}min".format(e+1,batch, len(p),loss_values / nrOfBatches,estimateOfArrival))

            # Validation
            trainTime = (time.time() - tic) / 60
            train_loss = loss_values / nrOfBatches

            p = valIndices

            tic = time.time()
            val_loss = 0
            nrOfBatches = 0.0
            self.model.save(self.checkpointPath+'/last')
            for batch in range(0, len(p), self.batchSize):
                input = convertToImage(inputs[p[batch:batch + self.batchSize]])
                if batch + self.batchSize > len(p):
                    nrOfBatches += (len(p) - batch) / self.batchSize
                else:
                    nrOfBatches += 1
                val_loss += self.loss(self.model(input,training=False),targets[p[batch:batch + self.batchSize],targetIndex])
            val_loss = tf.sqrt(val_loss / nrOfBatches)
            if val_loss < valLoss:
                valLoss = val_loss
                self.model.save(self.checkpointPath + '/best')
            print("val_score after epoch {}: {}".format(e+1,val_loss))
            inferenceTime = (time.time() - tic) / len(valIndices)
            if self.wandbProject != "":
                wandb.log({"train_loss": train_loss, "val_score": val_loss,
                           "train_time (min)": trainTime,
                           "epoch": e + 1,
                           "inference_time (s)": inferenceTime})



        #Test
        test_loss = 0
        nrOfBatches = 0.0
        for batch in range(0, len(testIndices), self.batchSize):
            input = convertToImage(inputs[testIndices[batch:batch + self.batchSize]])
            if batch + self.batchSize > len(p):
                nrOfBatches += (len(p) - batch) / self.batchSize
            else:
                nrOfBatches += 1
            test_loss += self.loss(self.model(input, training=False),
                                  targets[testIndices[batch:batch + self.batchSize], targetIndex])
        test_loss = tf.sqrt(test_loss / nrOfBatches)
        print("test_score: {}".format(test_loss))
        trainingTime = time.time() - trainingTime
        if self.wandbProject != "":
            wandb.log({"test_score": test_loss,"runtime": trainingTime})

        if self.wandbProject != "":
            run.finish()
        print("Total training time of {}h".format(trainingTime / 3600))

