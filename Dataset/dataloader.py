import os
import numpy as np
import logging
import torch
from sklearn import model_selection
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def data_loader(config):
    Data = numpy_loader(config)
    return Data

#data loading after preprocessing

def numpy_loader(config):
    # Build data
    Data = {}
    # Get the current directory path
    problem = config['problem']
    # Read the JSON file and load the data into a dictionary
    data_path = config['data_dir'] + '/' + problem + '/' + problem + '.npy'
    if os.path.exists(data_path):
        logger.info("Loading preprocessed " + problem)
        Data_npy = np.load(data_path, allow_pickle=True)
        if np.any(Data_npy.item().get('val_data')):
            Data['train_data'] = Data_npy.item().get('train_data')
            Data['train_label'] = Data_npy.item().get('train_label')
            Data['val_data'] = Data_npy.item().get('val_data')
            Data['val_label'] = Data_npy.item().get('val_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[1]
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy.item().get('train_data'), Data_npy.item().get('train_label'), 0.1)
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[2]

        Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
        Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
        logger.info("{} samples will be used for training".format(len(Data['All_train_data'])))
        samples, channels, time_steps = Data['All_train_data'].shape
        logger.info(
            "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    return Data


def split_dataset(data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label


class dataset_class(Dataset):
    def __init__(self, data, label, config):
        super(dataset_class, self).__init__()

        self.model_type = config['Model_Type'][0]
        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):
        x = self.feature[ind]
        x = x.astype(np.float32)
        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)

