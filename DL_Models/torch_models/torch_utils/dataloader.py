import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from config import config 
import logging 

def create_dataloader(X, y, batch_size, model_name, shuffle=True, drop_last=True,isClassification=False):
    """
    Input: X, y of type np.array
    Return: pytorch dataloader containing the dataset of X and y that returns batches of size batch_size 
    """
    # Transform np.array to torch flaot tensor
    tensor_x = torch.as_tensor(X).float()
    if isClassification:
        tensor_y = torch.as_tensor(y).long()
    else:
        tensor_y = torch.as_tensor(y).float()
    # Unsqueeze channel direction for eegNet model
    if model_name == 'EEGNet':
        logging.info(f"Unsqueeze data for eegnet")
        tensor_x = tensor_x.unsqueeze(1)
    # Log the shapes
    #logging.info(f"Tensor x {mode} size: {tensor_x.size()}")
    #logging.info(f"Tensor y {mode} size: {tensor_y.size()}")
    # Set device 
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    # Create dataset and dataloader 
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=1)