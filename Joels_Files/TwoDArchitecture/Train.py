from .dataLoader import loadData, split
from .architectures.resCNN import resCNN2D, CNN1D, resCNN3D
from .architectures.deep4 import Deep4Net, Shallow4Net, Hybrid4Net

from .architectures.CNN import CNN
from .architectures.PyramidalCNN import PyramidalCNN
from .architectures.eegNet import EEGNet
from .architectures.Xception import XCEPTION
from .architectures.Inception import INCEPTION

import os
from tensorflow import keras
import wandb
from tqdm import tqdm
import numpy as np
import time
import tensorflow as tf
from config import config
from .preprocess import convertToImage, convertToVideo
from sklearn.metrics import accuracy_score,mean_squared_error
from .updateModel import angle_loss
from utils.wandbHelper import getPredictionVisualisations
import sys

class method:

    def __init__(self, name:str = 'resCNN',directory: str = "./", imageShape:(int,int)=(32,32), nrOfSamples:int = 500,
                 batchSize: int = 64, wandbProject:str = "", continueTrainingBool: bool = False,
                 loss:str = 'mse', convDimension: int = 2, seed: int = 0, task: str = 'amplitude',
                 electrodes: np.ndarray = np.arange(1,130), dataPostFix: str = '', memoryEfficientBool: bool = True):
        config['framework'] = 'tensorflow'
        self.model = None
        self.name = name+'_{}D'.format(convDimension)
        self.batchSize = batchSize
        self.nrOfSamples = nrOfSamples
        self.checkpointPath = os.path.join(directory, self.name+"_{}_{}_{}_{}".format(task,str(len(electrodes)),dataPostFix,str(seed)))
        self.seed = seed
        self.convDimension = convDimension
        self.task = task
        self.patience = 20
        self.electrodes = electrodes - 1
        self.dataPostFix = dataPostFix
        self.memoryEfficientBool = memoryEfficientBool

        tf.random.set_seed(seed)
        np.random.seed(seed)
        if convDimension == 2:
            self.inputShape = (imageShape[0], imageShape[1],self.nrOfSamples)
            if name == "Deep4":
                self.preprocess = lambda x: x[...,np.newaxis]
                self.inversePreprocess = lambda x: x[..., 0]
                self.architecture = Deep4Net()
                self.inputShape = (500, len(electrodes), 1)
            elif name == "Shallow4":
                self.preprocess = lambda x: x[...,np.newaxis]
                self.inversePreprocess = lambda x: x[..., 0]
                self.architecture = Shallow4Net()
                self.inputShape = (500, len(electrodes), 1)
            elif name == "Hybrid4":
                self.preprocess = lambda x: x[..., np.newaxis]
                self.inversePreprocess = lambda x: x[..., 0]
                self.architecture = Hybrid4Net()
                self.inputShape = (500, len(electrodes), 1)
            else:
                self.preprocess = convertToImage
                self.architecture = resCNN2D(residualBool=True)
        elif convDimension == 3:
            self.preprocess = convertToVideo
            self.architecture = resCNN3D(residualBool=True)
            self.inputShape = (imageShape[0], imageShape[1], self.nrOfSamples, 1)
        elif convDimension == 12:
            self.preprocess = lambda x: np.expand_dims(x,axis=-1)
            self.inputShape = (500, len(electrodes), 1)
            self.architecture = resCNN2D(residualBool=True,kernelSize=5)
            self.batchSize = self.batchSize // 4
        elif convDimension == 1:
            if self.name == "EEGNet_1D":
                self.inputShape = (len(electrodes), 500)
                self.preprocess = lambda x: np.transpose(x, axes=(0, 2, 1))
                self.inversePreprocess = lambda x: np.transpose(x, axes=(0, 2, 1))
                self.architecture = EEGNet(F1 = 16, F2 = 256, D=4, kernel_size=256, dropout_rate = 0.5,
                norm_rate = 0.5, dropoutType = 'Dropout')
            elif self.name == "InceptionTime_1D":
                self.architecture = INCEPTION(kernelSize=64, convFilters=16, residualBool=True, depth=12, bottleneckSize=16)
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            elif self.name == "Xception_1D":
                self.architecture = XCEPTION(kernelSize=40, convFilters=64, residualBool=True, depth=18)
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            elif self.name == "CNN_1D":
                self.architecture = CNN(kernelSize=64, convFilters=16, residualBool=True, depth=12, regularization=0)
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            elif self.name == "PyramidalCNN_1D":
                self.architecture = PyramidalCNN(kernelSize=16, convFilters=16,residualBool=False, depth=6)
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            elif name == "Deep4":
                self.architecture = Deep4Net()
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            elif name == "Shallow4":
                self.architecture = Shallow4Net()
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            elif name == "Hybrid4":
                self.architecture = Hybrid4Net()
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            else:
                self.architecture = CNN1D()
                self.preprocess = lambda x: x
                self.inversePreprocess = lambda x: x
                self.inputShape = (500, len(electrodes))
            #self.architecture = PyramidalCNN(batch_size=batchSize,input_shape=self.inputShape)
        else:
            self.inputShape = (len(electrodes), 500)
            self.preprocess = lambda x: np.transpose(x,axes=(0,2,1))
            self.inversePreprocess = lambda x: np.transpose(x, axes=(0, 2, 1))
            if self.name == 'CNN_N':
                self.architecture = CNN1D(convFilters=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], kernelSize=64,
                                          maxPoolSize=9)
            else:
                self.architecture = CNN1D(kernelSize=5)

        self.learningRate = 0.0001
        self.wandbProject = wandbProject
        if self.task=='angle':
            loss = 'angle-loss'
        elif self.task=='lr':
            loss = "bce"



        if self.task == 'amplitude':
            self.targetIndex = 1
            self.outputShape = 1
            outputActivation = "linear"
            self.taskSet = "direction"
        elif self.task == 'angle':
            self.targetIndex = 2
            self.outputShape = 1
            outputActivation = "linear"
            self.taskSet = "direction"
        elif self.task == "lr":
            self.targetIndex = 1
            self.outputShape = 1
            outputActivation = "sigmoid"
            self.taskSet = "lr"
        elif self.task == "position":
            self.targetIndex = [1,2]
            self.outputShape = 2
            outputActivation = "linear"
            self.taskSet = "position"
        else:
            print("Not a valid task.")
            self.targetIndex = [1,2]
            self.outputShape = 2
            outputActivation = "linear"
            self.taskSet = ""

        if loss=='angle-loss':
            self.score = (lambda y_pred, y: np.sqrt(angle_loss(y, y_pred.ravel())))
            self.lossForFit = angle_loss
            self.lossName = loss
        elif loss == "bce":
            self.score = (lambda y_pred, y: accuracy_score(y, np.rint(y_pred.ravel())))
            self.lossForFit = 'binary_crossentropy'
            self.lossName = 'binary_crossentropy'
        else:
            if self.task == "amplitude":
                self.score = (lambda y_pred, y: np.sqrt(mean_squared_error(y, y_pred.ravel())))
            else:
                self.score = (lambda y_pred, y: np.linalg.norm(y - y_pred, axis=1).mean())
            self.lossForFit = 'mse'
            self.lossName = 'mse'


        self.model = self.architecture.buildModel(inputShape=self.inputShape, outputShape = self.outputShape,
                                                  loss=self.lossForFit,outputActivation=outputActivation)

        if not os.path.exists(self.checkpointPath):
            os.mkdir(self.checkpointPath)
        elif continueTrainingBool:
            try:
                self.model = keras.models.load_model(self.checkpointPath)
            except:
                pass


    def fit(self, nrOfEpochs: int = 50, saveBool: bool = True):
        if self.wandbProject != "":
            run = wandb.init(project=self.wandbProject, entity='deepeye')
            wandb.run.name = self.name
            wandbConfig = wandb.config
        inputPath = config['data_dir'] + "{}_{}/".format(self.taskSet,self.dataPostFix) + 'X.npy'
        targetPath = config['data_dir'] + "{}_{}/".format(self.taskSet,self.dataPostFix) + 'Y.npy'
        if self.memoryEfficientBool:
            mmapMode = 'c'
        else:
            mmapMode = None
        inputs, targets = loadData(inputPath,targetPath,mmapMode=mmapMode)
        trainIndices, valIndices, testIndices = split(targets[:,0], 0.7, 0.15, 0.15)
        targets = targets[:,self.targetIndex]
        if len(self.electrodes) != 129:
            inputs = inputs[:,:,self.electrodes]
        self.model.summary()

        trainable_count = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        non_trainable_count = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        nr_params = trainable_count + non_trainable_count
        if self.wandbProject != "":
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            wandbConfig.update({"Directory": self.checkpointPath, "Learning_Rate": self.learningRate,
                           "Input_Shape": "{}".format(','.join([str(s) for s in self.inputShape])),
                                "Training Set": inputPath, "Model_Name": self.name, "Task": self.task,
                                "Type": self.convDimension, "Batch_Size": self.batchSize,
                                "Trainable_Params": trainable_count,
                                "Non_Trainable_Params": non_trainable_count,
                                "Nr_Of_Params": nr_params, "Model": stringlist,
                                "Electrodes": np.array2string(self.electrodes+1)})

        pbar = tqdm(range(nrOfEpochs))
        valLoss = np.inf
        trainingTime = time.time()
        best_val_loss = sys.maxsize
        patience = 0

        for e in pbar:
            loss_values = 0
            tic = time.time()

            # Permute
            p = np.random.permutation(len(trainIndices))

            nrOfBatches = 0.0
            totNrOfBatches = len(p) / self.batchSize
            for batch in range(0, len(p), self.batchSize):
                input = self.preprocess(inputs[p[batch:batch + self.batchSize]])
                target = targets[p[batch:batch + self.batchSize]].astype(np.float32)
                loss_values_temp = self.model.train_on_batch(tf.convert_to_tensor(input), tf.convert_to_tensor(target))
                loss_values += loss_values_temp
                if batch + self.batchSize > len(p):
                    nrOfBatches += (len(p) - batch) / self.batchSize
                else:
                    nrOfBatches += 1
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
            if saveBool:
                self.model.save(self.checkpointPath+'/last')
            for batch in range(0, len(p), self.batchSize):
                input = self.preprocess(inputs[p[batch:batch + self.batchSize]])
                if batch + self.batchSize > len(p):
                    nrOfBatches += (len(p) - batch) / self.batchSize
                else:
                    nrOfBatches += 1
                val_loss += self.model.test_on_batch(input,targets[p[batch:batch + self.batchSize]])
            val_loss = np.sqrt(val_loss / nrOfBatches)
            if val_loss < valLoss:
                valLoss = val_loss
                if saveBool:
                    self.model.save(self.checkpointPath + '/best')
            if self.wandbProject == "":
                print("val_score after epoch {}: {}".format(e+1,val_loss))
            inferenceTime = (time.time() - tic) / len(valIndices)
            if self.wandbProject != "":
                wandb.log({"train_loss": train_loss, "val_score": val_loss,
                           "train_time (min)": trainTime,
                           "epoch": e + 1,
                           "inference_time (s)": inferenceTime})

            if patience > self.patience:
                print("Early Stop after {} Epochs.".format(e+1))
                break
            if val_loss >= best_val_loss:
                patience += 1
            else:
                best_val_loss = val_loss
                patience = 0



        #Test
        nrOfBatches = 0.0
        predictions = np.zeros([len(testIndices),self.outputShape])
        for batch in range(0, len(testIndices), self.batchSize):
            input = self.preprocess(inputs[testIndices[batch:batch + self.batchSize]])
            if batch + self.batchSize > len(p):
                nrOfBatches += (len(p) - batch) / self.batchSize
            else:
                nrOfBatches += 1

            predictions[batch:batch + self.batchSize] = self.model.predict(input,verbose=0)
        test_loss = self.score(predictions, targets[testIndices])
        print("test_score: {}".format(test_loss))
        trainingTime = time.time() - trainingTime
        if self.wandbProject != "":
            logs = {"test_score": test_loss,"runtime": trainingTime}
            #prediction = self.model.predict(self.preprocess(inputs[[testIndices[:16]]]),verbose=0)
            #addLogs = getPredictionVisualisations(self.model,self.name,inputs[[testIndices[:16]]],
            #                                      targets[[testIndices[:16]]],prediction,self.lossName,
            #                                      preprocess=self.preprocess,inversePreprocess=self.inversePreprocess)
            #wandb.log({**logs, **addLogs})
            wandb.log(logs)

        if self.wandbProject != "":
            run.finish()
        print("Total training time of {}min".format(trainingTime / 60))

