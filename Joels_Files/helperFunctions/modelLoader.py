import os
from config import config

def returnTorchModel(path: str):
    architectureList = ['CNN', 'EEGNet', 'InceptionTime', 'PyramidalCNN', 'Xception', 'biLSTM', 'GCN', 'LSTM', 'UNet'
        , 'TransformerSimple', 'ConvLSTM']
    originalLength = len(architectureList)
    architectureList = architectureList + [['_angle' + sub for sub in architectureList]] + [
        ['_amplitude' + sub for sub in architectureList]]
    basename = os.path.basename(path)
    model = None
    modelName = None
    modelType = None
    for i, a in enumerate(architectureList):
        if basename.lower().startswith(a.lower()):
            type = i - int(i / originalLength) * originalLength
            typeList = [None,'angle','amplitude']
            model = loadModelTorch(name=architectureList[type])
            modelName = architectureList[type]
            modelType = typeList[type]
    from hyperparameters import all_models
    params = all_models[config['task']][config['dataset']][config['preprocessing']]
    if modelType is not None:
        params = all_models[config['task']][config['dataset']][config['preprocessing']][modelType]
    return model(path=path, **params[modelName])

def loadModelTorch(name:str):
    if name == 'CNN':
        from DL_Models.torch_models.CNN.CNN import CNN
        return CNN
    elif name == 'EEGNet':
        from DL_Models.torch_models.EEGNet.eegNet import EEGNet
        return EEGNet
    elif name == 'InceptionTime':
        from DL_Models.torch_models.InceptionTime.InceptionTime import Inception
        return Inception
    elif name == 'PyramidalCNN':
        from DL_Models.torch_models.PyramidalCNN.PyramidalCNN import PyramidalCNN
        return PyramidalCNN
    elif name == 'Xception':
        from DL_Models.torch_models.Xception.Xception import XCEPTION
        return XCEPTION
    # elif name == 'biLSTM':
    #     from DL_Models.torch_models.BiLSTM.biLSTM import biLSTM
    #     return biLSTM
    # elif name == 'GCN':
    #     from DL_Models.torch_models.GCN.GCN import GCN
    #     return GCN
    # elif name == 'LSTM':
    #     from DL_Models.torch_models.LSTM.LSTM import LSTM
    #     return LSTM
    # elif name == 'UNet':
    #     from DL_Models.torch_models.UNet.UNet import UNet
    #     return UNet
    # elif name == 'TransformerSimple':
    #     from DL_Models.torch_models.Transformer.TransformerSimple import TransformerSimple
    #     return TransformerSimple
    # elif name == 'ConvLSTM':
    #     from DL_Models.torch_models.ConvLSTM.ConvLSTM import ConvLSTM
    #     return ConvLSTM
    else:
        raise Exception("choose valid model")