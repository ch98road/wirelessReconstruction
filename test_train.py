from lib2to3.pytree import convert
import numpy as np
# import h5py
import torch
from Model_define_pytorch import NMSE, AutoEncoder, DatasetFolder
import os
import torch.nn as nn
import config.config as cfg
from torchvision.utils import save_image
from PIL import Image
from utils.Loss import NMSE_LOSS,NMSE_POWER

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = 1
epochs = 1
learning_rate = 1e-3
num_workers = 4
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 512

model = AutoEncoder(feedback_bits)
if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()
import scipy.io as scio

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
input = torch.rand(1, 2, 126, 128).cuda()
# input2 = torch.rand(1, 2, 126, 128)
# input = torch.rand(256, 2, 126, 128).cuda()
output = model(input)
loss = criterion(output * 256, input * 256)
power = NMSE_POWER(output)
loss2 = NMSE_LOSS(output=output,input=input)
# loss /= power
loss += loss2
optimizer.zero_grad()
# # loss.backward()
loss.backward()
optimizer.step()
