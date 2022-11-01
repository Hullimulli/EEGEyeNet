import numpy as np
import tensorflow as tf

class baselineBuilder:
    def __init__(self, predType='mean'):
        pass
    def buildModel(self,inputShape=None, outputShape=None, loss=None, outputActivation=None):
        return meanPredictor(outputShape,loss)
class meanPredictor:

    def __init__(self,outputShape,loss):
        self.seenSamples = 0
        self.total = np.zeros(outputShape)
        self.trainable_weights = tf.convert_to_tensor([0])
        self.non_trainable_weights = tf.convert_to_tensor([0])
        self.loss = lambda x,y: np.mean(np.square(x-y))

    def summary(self, print_fn=None):
        print("Mean_Predictor")
        return "Mean_Predictor"

    def train_on_batch(self,input=None,target=None):
        target = target.numpy()
        self.seenSamples += target.shape[0]
        self.total +=  np.sum(target,axis=0)

        return np.mean(self.loss(self.predict(input), target))

    def test_on_batch(self,input=None,target=None):
        return np.mean(self.loss(self.predict(input), target))

    def save(self,path=None):
        pass

    def predict(self, input=None, verbose=None):
        prediction = self.total / self.seenSamples
        axes = np.zeros(1 + prediction.ndim, dtype=np.int64) + 1
        axes[0] = input.shape[0]
        prediction = np.tile(prediction[np.newaxis],axes)
        return prediction