import numpy as np
from config import config
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d

def saliencyMap(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, normalizeBool: bool = True, includeInputBool: bool = False) -> np.ndarray:
    if config['framework'] == 'tensorflow':
        return saliencyMapTensorflow(model, inputSignals, groundTruth, loss, normalizeBool, includeInputBool)
    elif config['framework'] == 'pytorch':
        return saliencyMapTorch(model, inputSignals, groundTruth, loss, normalizeBool, includeInputBool)
    else:
        raise Exception("SaliencyMap not available for framework.")


def fullGrad(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, normalizeBool: bool = True, biasOnlyBool: bool = False) -> np.ndarray:
    if config['framework'] == 'tensorflow':
        return fullGradTensorflow(model, inputSignals, groundTruth, loss, normalizeBool, biasOnlyBool)
    elif config['framework'] == 'pytorch':
        return fullGradTorch(model, inputSignals, groundTruth, loss, normalizeBool, biasOnlyBool)
    else:
        raise Exception("SaliencyMap not available for framework.")

def saliencyMapTensorflow(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str , normalizeBool: bool = True, includeInputBool: bool = False) -> np.ndarray:
    import tensorflow as tf

    if loss == "bce":
        loss = lambda y,yPred: tf.multiply(-2*(y-0.5),yPred)
    elif loss == "mse":
        loss = lambda y,yPred: tf.norm(y - yPred, axis=1)
    elif loss == "angle-loss":
        loss = lambda y,yPred: tf.sqrt(tf.square(tf.math.atan2(tf.sin(yPred - y), tf.cos(yPred - y))))

    inputSignals = tf.convert_to_tensor(inputSignals)

    with tf.GradientTape() as g:
        g.watch(inputSignals)
        outputs = model(inputSignals,training=False)
        lossValues = loss(groundTruth,outputs)

    grads = g.gradient(lossValues,inputSignals)

    grads = grads.numpy()

    if includeInputBool:
        grads = grads * inputSignals
    grads = np.abs(grads)

    if normalizeBool:
        for i in range(grads.shape[0]):
            if np.max(grads[i]) != 0:
                grads[i] = grads[i] / np.max(grads[i])


    return grads


def saliencyMapTorch(model, inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, normalizeBool: bool = True, includeInputBool: bool = False) -> np.ndarray:
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.double()
    if loss == "bce":
        loss = lambda y,yPred: torch.multiply(-2*(y-0.5),yPred)
    elif loss == "mse":
        loss = lambda y,yPred: torch.norm(y - yPred, dim=1)
    elif loss == "angle-loss":
        loss = lambda y,yPred: torch.sqrt(torch.square(torch.atan2(torch.sin(yPred - y), torch.cos(yPred - y))))
    grads = np.zeros(inputSignals.shape)
    inputSignals = np.transpose(inputSignals, (0, 2, 1))
    groundTruth = torch.from_numpy(groundTruth).double().to(device)
    inputSignals = torch.from_numpy(inputSignals).double().to(device)
    for i in range(inputSignals.shape[0]):
        temp = inputSignals[[i]].requires_grad_()
        outputs = model(temp)
        lossValues = loss(groundTruth[i], outputs)
        lossValues.backward()
        grads[i] = np.transpose(temp.grad.data.cpu().detach().numpy(), (0, 2, 1))
        if includeInputBool:
            grads[i] = grads[i] * np.transpose(inputSignals[i].cpu().detach().numpy())
        grads[i] = np.abs(grads[i])
    if normalizeBool:
        for i in range(grads.shape[0]):
            if np.max(grads[i]) != 0:
                grads[i] = grads[i] / np.max(grads[i])

    return grads


def fullGradTensorflow(model,inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, normalizeBool: bool = True, biasOnlyBool: bool = False) -> np.ndarray:
    import tensorflow.keras as keras
    import tensorflow as tf


    if loss == "bce":
        loss = lambda y,yPred: tf.multiply(-2*(y-0.5),yPred)
    elif loss == "mse":
        loss = lambda y,yPred: tf.norm(y - yPred, axis=1)
    elif loss == "angle-loss":
        loss = lambda y,yPred: tf.sqrt(tf.square(tf.math.atan2(tf.sin(yPred - y), tf.cos(yPred - y))))

    map = np.zeros(inputSignals.shape)
    biasMap = np.zeros(inputSignals.shape)
    nrOfBiases = 0

    def watch_layer(layer, tape):
        """
        Make an intermediate hidden `layer` watchable by the `tape`.
        After calling this function, you can obtain the gradient with
        respect to the output of the `layer` by calling:

            grads = tape.gradient(..., layer.result)

        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Store the result of `layer.call` internally.
                layer.result = func(*args, **kwargs)
                # From this point onwards, watch this tensor.
                tape.watch(layer.result)
                # Return the result to continue with the forward pass.
                return layer.result

            return wrapper

        layer.call = decorator(layer.call)
        return layer
    inputSignals = tf.convert_to_tensor(inputSignals)
    for i in range(len(model.layers)):
        b = None
        grads = None
        if "batch" in model.layers[i].get_config()['name'].lower():
            layerOfInterest = model.get_layer(model.layers[i].get_config()['name'].lower())
            with tf.GradientTape(persistent=True) as g:
                watch_layer(layerOfInterest,g)
                output = model(inputSignals,training=False)
                lossValues = loss(groundTruth, output)
            b = g.gradient(lossValues,layerOfInterest.result)
            bias = - (model.layers[i].moving_mean * model.layers[i].gamma
                   / tf.sqrt(model.layers[i].moving_variance + model.layers[i].epsilon)) + model.layers[i].beta
            b = keras.backend.abs(np.expand_dims(bias,axis=(0,1)) * b)
            b = b.numpy()
        elif "conv" in model.layers[i].get_config()['name'].lower() and model.layers[i].bias is not None:
            with tf.GradientTape(persistent=True) as g:
                g.watch(model.layers[i].bias)
                outputs = model(inputSignals, training=False)
                lossValues = loss(groundTruth, outputs)
            b = g.gradient(lossValues, model.layers[i].bias)
            bias = np.atleast_3d(model.layers[i].bias)
            b = keras.backend.abs(b * bias)
            b = b.numpy()
        elif i == len(model.layers)-1:
            with tf.GradientTape(persistent=True) as g:
                g.watch(inputSignals)
                outputs = model(inputSignals, training=False)
                lossValues = loss(groundTruth, outputs)
            grads = g.gradient(lossValues, inputSignals)
            grads = keras.backend.abs(grads * inputSignals)
            grads = grads.numpy()

        if b is not None:
            b = b / np.amax(b, axis=-1, keepdims=True)
            biasMean = np.nanmean(b, axis=-1, keepdims=True)
            biasMean[np.isnan(biasMean)] = 0
            if biasMean.shape[1] != biasMap.shape[1]:
                x = np.arange(biasMean.shape[1]+1)
                f = interp1d(x,np.concatenate((biasMean[0,:,0],biasMean[0,[-1],0])))
                xnew = np.arange(0,biasMean.shape[1],biasMean.shape[1] / biasMap.shape[1])
                biasMean = f(xnew)
            biasMap += np.atleast_3d(biasMean)
            nrOfBiases += 1
        if grads is not None:
            del g
            lmd = 1
            if biasOnlyBool:
                map = biasMap / nrOfBiases
            else:
                map = grads / np.max(grads) + lmd * biasMap / nrOfBiases

    if normalizeBool:
        for i in range(map.shape[0]):
            map[i] = map[i] / np.max(map[i])

    return map



def fullGradTorch(model,inputSignals: np.ndarray, groundTruth: np.ndarray, loss: str, normalizeBool: bool = True, biasOnlyBool: bool = False) -> np.ndarray:
    import torch
    import torch.nn as nn

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.double()
    if loss == "bce":
        loss = lambda y, yPred: torch.multiply(-2 * (y - 0.5), yPred)
    elif loss == "mse":
        loss = lambda y, yPred: torch.norm(y - yPred, dim=1)
    elif loss == "angle-loss":
        loss = lambda y, yPred: torch.sqrt(torch.square(torch.atan2(torch.sin(yPred - y), torch.cos(yPred - y))))

    map = np.zeros(inputSignals.shape)
    biasMap = np.zeros(map.shape)
    inputSignals = np.transpose(inputSignals, (0, 2, 1))
    groundTruth = torch.from_numpy(groundTruth).double().to(device)
    inputSignals = torch.from_numpy(inputSignals).double().to(device)

    featureGrads = []
    biases = []
    handles = []

    def extractLayerGrads(module, in_grad, out_grad):
        # function to collect the gradient outputs
        # from each layer

        if not module.bias is None:
            featureGrads.append(out_grad[0])

    def extractLayerBias(module):
        # extract bias of each layer

        # for batchnorm, the overall "bias" is different
        # from batchnorm bias parameter.
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b)
        # Thus (-m*w/s + b) is the effective bias of batchnorm

        if isinstance(module, nn.BatchNorm2d):
            b = - (module.running_mean * module.weight
                    / torch.sqrt(module.running_var + module.eps)) + module.bias
            return b.data
        elif module.bias is None:
            return None
        else:
            return module.bias.data

    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
            handleG = m.register_full_backward_hook(extractLayerGrads)
            handles.append(handleG)

            b = extractLayerBias(m)
            if (b is not None): biases.append(b)

    for input in range(inputSignals.size(dim=0)):
        temp = inputSignals[[input]].requires_grad_()
        outputs = model(temp)
        lossValues = loss(groundTruth[[input]], outputs)
        lossValues.backward()
        map[[input]] = np.transpose(temp.grad.data.cpu().detach().numpy(), (0, 2, 1))

        map[[input]] = np.absolute(map[[input]] * np.transpose(inputSignals[[input]].cpu().detach().numpy(), (0, 2, 1)))
        nrOfBiases = 0
        for i in range(len(featureGrads)):
            # Select only Conv layers
            if len(featureGrads[i].shape) == len(inputSignals.shape):
                intermedGrad = np.transpose(featureGrads[i].cpu().detach().numpy(), (0, 2, 1))
                intermedGrad = np.abs(intermedGrad)
                intermedGrad /= np.max(intermedGrad,axis=-1,keepdims=True)
                intermedGrad = np.nanmean(intermedGrad,axis=-1,keepdims=True)
                intermedGrad[np.isnan(intermedGrad)] = 0
                x = np.arange(intermedGrad.shape[1] + 1)
                f = interp1d(x, np.concatenate((intermedGrad[0, :, 0], intermedGrad[0, [-1], 0])))
                xnew = np.arange(0, intermedGrad.shape[1], intermedGrad.shape[1] / biasMap.shape[1])
                intermedGrad = f(xnew)
                biasMap[[input]] += np.atleast_3d(intermedGrad)
                nrOfBiases+=1
        lmd = 1
        if biasOnlyBool:
            map[[input]] = biasMap[[input]] / nrOfBiases
        else:
            map[[input]] = map[[input]] / np.max(map[[input]]) + lmd * biasMap[[input]] / nrOfBiases

    if normalizeBool:
        for i in range(map.shape[0]):
            map[i] = map[i] / np.max(map[i])

    return map

def plotSaliencyMap(inputSignals: np.ndarray, groundTruth: np.ndarray, gradients: np.ndarray, directory: str,
                  electrodesUsedForTraining: np.ndarray = np.arange(1, 130),electrodesToPlot: np.ndarray = np.arange(1, 130),
                  filename: str = 'AttentionVisualisation', format: str = 'pdf',maxValue: float = 100, saveBool: bool = True):

    indicesToPlot = findElectrodeIndices(electrodesUsedForTraining,electrodesToPlot)

    linSpace = np.arange(1, inputSignals.shape[1]*2+1, 2)
    for i in range(inputSignals.shape[0]):
        for j in indicesToPlot:
            f, ax = plt.subplots()
            ax.title.set_text(
                "Electrode {} of Sample {}".format(electrodesUsedForTraining[j], str(i)) + ", GroundTruth: " + np.array2string(
                    groundTruth[i]))
            ax.set_ylim([-maxValue, maxValue])
            ax.get_xaxis().set_major_formatter(FormatStrFormatter('%d ms'))
            ax.get_yaxis().set_major_formatter(FormatStrFormatter('%d mv'))
            ax.plot(linSpace, inputSignals[i, :, j], c='black')
            ax.imshow(np.repeat(np.repeat(np.expand_dims(gradients[i, :, j], axis=0), 2, axis=1), int(maxValue * 2), axis=0),
                      extent=[0, 1000, int(maxValue), -int(maxValue)], cmap='jet', alpha=0.5, origin='lower')
            if saveBool:
                plt.savefig(os.path.join(directory,
                                         filename + '_Sample{}_El{}.'.format(str(i), electrodesUsedForTraining[j]) + format))
            else:
                return plt
            plt.close()



###############Helper Functions#####################

def findElectrodeIndices(electrodesUsedForTraining: np.ndarray = np.arange(1, 130),
                         electrodesToPlot: np.ndarray = np.arange(1, 130)) -> np.ndarray:
    intersect, ind_a, electrodes = np.intersect1d(electrodesToPlot, electrodesUsedForTraining, return_indices=True)
    del intersect, ind_a

    return np.atleast_1d(electrodes)