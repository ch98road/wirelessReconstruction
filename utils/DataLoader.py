from unittest import loader
from mim import train
import scipy.io as scio
import numpy as np
from Model_define_pytorch import DatasetFolder
import torch
import os


def DataLoader(
        datapath='/work/workspace3/chenhuan/code/wirelessReconstruction/data/Htrain.mat',
        datatype='train',
        batch_size=128,
        num_workers=4,
        distributed=False):
    mat = scio.loadmat(datapath)
    if datatype == 'train':
        data = mat['H_train']  # shape=8000*126*128*2
        shuffle = True
    elif datatype == 'test':
        data = mat['H_test']  # shape=2000*126*128*2
        shuffle = False
    else:
        raise 'Wrong datatype'
    print(np.shape(data))
    data = np.transpose(data.astype('float32'), [0, 3, 1, 2])
    dataset = DatasetFolder(data)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             sampler=sampler)
    else:
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=True)

    return data, dataset, loader


def ReadData(
        datapath='/work/workspace3/chenhuan/code/wirelessReconstruction/data/Htrain.mat',
        datatype='train'):
    mat = scio.loadmat(datapath)
    if datatype == 'train':
        data = mat['H_train']  # shape=8000*126*128*2
    elif datatype == 'test':
        data = mat['H_test']  # shape=2000*126*128*2
    return data


def getMeanStd(dataroot='/work/workspace3/chenhuan/code/wirelessReconstruction/data'):
    train_path = os.path.join(dataroot, 'Htrain.mat')
    test_path = os.path.join(dataroot, 'Htest.mat')

    train_data = ReadData(datapath=train_path, datatype='train')
    test_data = ReadData(datapath=test_path, datatype='test')

    cat_data = np.concatenate([train_data, test_data], axis=0)
    mean = np.mean(cat_data)
    std = np.std(cat_data)

    return mean, std
