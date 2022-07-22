import os
from config import config
from hyperparameters import batch_size, all_models

def returnTorchModel(path: str, loss: str = 'mse'):
    import torch
    architectureList = ['CNN', 'EEGNet', 'InceptionTime', 'PyramidalCNN', 'Xception', 'biLSTM', 'GCN', 'LSTM', 'UNet'
        , 'TransformerSimple', 'ConvLSTM']
    architectureList = architectureList
    basename = os.path.basename(path)
    model = None
    modelName = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for i, a in enumerate(architectureList):
        if a.lower() in basename.lower():
            modelName = a
            model = loadModelTorch(name=architectureList[i])

    params = all_models[config['task']][config['dataset']][config['preprocessing']]
    if config['task'] == 'Direction_task':
        if loss == 'angle-loss':
            params = params['angle']
        else:
            params = params['amplitude']
    params = params[modelName][1]
    entriesToRemove = ['nb_models']
    for k in entriesToRemove:
        params.pop(k, None)

    model = model(path=path,model_number=-1, **params)
    model.load_state_dict(torch.load(path,map_location=device))
    return model

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
    elif name == 'biLSTM':
        from DL_Models.torch_models.BiLSTM.biLSTM import biLSTM
        return biLSTM
    elif name == 'GCN':
        from DL_Models.torch_models.GCN.GCN import GCN
        return GCN
    elif name == 'LSTM':
        from DL_Models.torch_models.LSTM.LSTM import LSTM
        return LSTM
    elif name == 'UNet':
        from DL_Models.torch_models.UNet.UNet import UNet
        return UNet
    elif name == 'TransformerSimple':
        from DL_Models.torch_models.Transformer.TransformerSimple import TransformerSimple
        return TransformerSimple
    elif name == 'ConvLSTM':
        from DL_Models.torch_models.ConvLSTM.ConvLSTM import ConvLSTM
        return ConvLSTM
    elif name == 'CNNMultiTask':
        from DL_Models.torch_models.CNN.CNNMultiTask import CNNMultiTask
        return CNNMultiTask
    else:
        raise Exception("choose valid model")