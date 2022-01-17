#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder
import os
import torch.nn as nn

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = 256
epochs = 1000
learning_rate = 1e-3
num_workers = 4
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 512

from adversarial import Adversarial
ganloss = Adversarial()
# Model construction
model = AutoEncoder(feedback_bits)
x_train = torch.randn(1,126,128,2)
x_train = np.transpose(x_train,[0,3,1,2])
output = model(x_train)
print(x_train.shape, output.shape)
# for i in range(10):
loss_adv = ganloss(x_train, output)
