from .dataLoader import loadData, split
from .resCNN import resCNN2D, CNN1D, resCNN3D
from .PyramidalCNN import PyramidalCNN
import os
import tensorflow.keras as keras
import wandb
from tqdm import tqdm
import numpy as np
import time
import tensorflow as tf
from config import config
from .preprocess import convertToImage, convertToVideo
from sklearn.metrics import mean_squared_error
from .updateModel import mseUpdate, angleLossUpdate, mse, angle_loss
from utils.wandbHelper import getPredictionVisualisations

class method:

    def __init__(self, name:str = 'resCNN',directory: str = "./", imageShape:(int,int)=(32,32), nrOfSamples:int = 500,
                 batchSize: int = 32, wandbProject:str = "", continueTrainingBool: bool = False,
                 loss:str = 'mse', convDimension: int = 2, seed: int = 0, task: str = 'amplitude'):
        config['framework'] = 'tensorflow'
        self.model = None
        self.name = name+'_{}D'.format(convDimension)
        self.batchSize = batchSize
        self.nrOfSamples = nrOfSamples
        self.checkpointPath = os.path.join(directory, self.name)
        self.seed = seed
        self.convDimension = convDimension
        self.task = task
        tf.random.set_seed(seed)
        np.random.seed(seed)
        if convDimension == 2:
            self.inputShape = (imageShape[0], imageShape[1],self.nrOfSamples)
            self.preprocess = convertToImage
            self.architecture = resCNN2D(residualBool=True)
        elif convDimension == 3:
            self.preprocess = convertToVideo
            self.architecture = resCNN3D(residualBool=True)
            self.inputShape = (imageShape[0], imageShape[1], self.nrOfSamples, 1)
        elif convDimension == 1:
            self.inputShape = (500,129)
            if self.name == 'CNN_N':
                self.architecture = CNN1D(convFilters=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], kernelSize=64,
                                          maxPoolSize=9)
            else:
                self.architecture = CNN1D()
            #self.architecture = PyramidalCNN(batch_size=batchSize,input_shape=self.inputShape)
            self.preprocess = lambda x: x
            self.inversePreprocess = lambda x: x
        else:
            self.inputShape = (129, 500)
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

        if loss=='angle-loss':
            self.update = angleLossUpdate
            self.loss = angle_loss
            self.lossForFit = angle_loss
            self.lossName = loss
        else:
            self.update = mseUpdate
            self.loss = (lambda y_pred, y: mean_squared_error(y, y_pred.ravel()))
            self.lossForFit = 'mse'
            self.lossName = 'mse'

        self.model = self.architecture.buildModel(inputShape=self.inputShape, loss=self.lossForFit)
        if not os.path.exists(self.checkpointPath):
            os.mkdir(self.checkpointPath)
        elif continueTrainingBool:
            try:
                self.model = keras.models.load_model(self.checkpointPath)
            except:
                pass


    def fit(self, nrOfEpochs: int = 50, saveBool: bool = True):
        if self.wandbProject != "":
            run = wandb.init(project=self.wandbProject, entity='hullimulli')
            wandb.run.name = self.name
            wandbConfig = wandb.config
        inputPath = config['data_dir'] + config['all_EEG_file'][:-4] + '_X.npy'
        targetPath = config['data_dir'] + config['all_EEG_file'][:-4] + '_Y.npy'
        if self.task == 'amplitude':
            targetIndex = 1
        elif self.task == 'angle':
            targetIndex = 2
        else:
            targetIndex = [1,2]
        inputs, targets = loadData(inputPath,targetPath)
        trainIndices, valIndices, testIndices = split(targets[:,0], 0.7, 0.15, 0.15)
        targets = targets[:,targetIndex]
        if self.model is None:
            self.model = self.architecture.buildModel(inputShape=self.inputShape, loss=self.lossForFit)

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
                                "Nr_Of_Params": nr_params, "Model": stringlist})

        pbar = tqdm(range(nrOfEpochs))
        valLoss = np.inf
        trainingTime = time.time()
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
                if self.task == 'angle':
                    loss_values_temp = self.model.train_on_batch(tf.convert_to_tensor(input), tf.convert_to_tensor(target))
                else:
                    loss_values_temp = self.update(model=self.model, input=tf.convert_to_tensor(input),
                                                        ground=tf.convert_to_tensor(target), seed=tf.convert_to_tensor(self.seed+batch))
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
                if self.task == 'angle':
                    val_loss += self.model.test_on_batch(input,targets[p[batch:batch + self.batchSize]])
                else:
                    val_loss += self.loss(self.model(input,training=False).numpy(),targets[p[batch:batch + self.batchSize]])
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



        #Test
        test_loss = 0
        nrOfBatches = 0.0
        for batch in range(0, len(testIndices), self.batchSize):
            input = self.preprocess(inputs[testIndices[batch:batch + self.batchSize]])
            if batch + self.batchSize > len(p):
                nrOfBatches += (len(p) - batch) / self.batchSize
            else:
                nrOfBatches += 1
            if self.task == 'angle':
                test_loss += self.model.test_on_batch(input, targets[testIndices[batch:batch + self.batchSize]])
            else:
                test_loss += self.loss(self.model(input, training=False).numpy(),
                                  targets[testIndices[batch:batch + self.batchSize]])
        test_loss = np.sqrt(test_loss / nrOfBatches)
        print("test_score: {}".format(test_loss))
        trainingTime = time.time() - trainingTime
        if self.wandbProject != "":
            logs = {"test_score": test_loss,"runtime": trainingTime}
            prediction = self.model.predict(self.preprocess(inputs[[testIndices[:16]]]))
            addLogs = getPredictionVisualisations(self.model,self.name,inputs[[testIndices[:16]]],
                                                  targets[[testIndices[:16]]],prediction,self.lossName,
                                                  preprocess=self.preprocess,inversePreprocess=self.inversePreprocess)
            wandb.log({**logs, **addLogs})

        if self.wandbProject != "":
            run.finish()
        print("Total training time of {}min".format(trainingTime / 60))

