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
# import h5py
import torch
from model.Model_define_pytorch_office import AutoEncoder, DatasetFolder
from model.adversarial import Adversarial

import os
import torch.nn as nn
import config.config_officeWithL2 as cfg
import utils.CosineAnnealingWithWarmup as LR
import scipy.io as scio
from utils.Loss import l2_regularization, l2_regularization_loss
from tensorboardX import SummaryWriter

if not os.path.exists('{}/log/{}'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH)):
    os.mkdir('{}/log/{}'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))
if not os.path.exists('{}/{}'.format(cfg.PROJECT_ROOT, cfg.MODEL_SAVE_PATH)):
    os.mkdir('{}/{}'.format(cfg.PROJECT_ROOT, cfg.MODEL_SAVE_PATH))

writer = SummaryWriter('{}/log/{}/'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
use_single_gpu = True  # select whether using single gpu or multiple gpus
# use_single_gpu = False  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
batch_size = 256
epochs = 1
learning_rate = 1e-3
num_workers = 4
print_freq = 10  # print frequency (default: 60)
# parameters for data
feedback_bits = 512
Pretrain = True
l2_alpha = 0.2

# Model construction
model = AutoEncoder(feedback_bits)
if Pretrain:
    model.encoder.load_state_dict(
        torch.load(cfg.PRE_ENCODER_PATH)['state_dict'])
    model.decoder.load_state_dict(
        torch.load(cfg.PRE_DECODER_PATH)['state_dict'])
    print("weight loaded")
if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
data_load_address = '{}/data'.format(cfg.PROJECT_ROOT)
mat = scio.loadmat(data_load_address + '/Htrain.mat')
x_train = mat['H_train']  # shape=8000*126*128*2

x_train = np.transpose(x_train.astype('float32'), [0, 3, 1, 2])
print(np.shape(x_train))
mat = scio.loadmat(data_load_address + '/Htest.mat')
x_test = mat['H_test']  # shape=2000*126*128*2

x_test = np.transpose(x_test.astype('float32'), [0, 3, 1, 2])
print(np.shape(x_test))
# Data loading

# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           pin_memory=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=True)
best_loss = 1
cosineLR = LR.LR_Scheduler(optimizer=optimizer,
                           warmup_epochs=5,
                           warmup_lr=learning_rate,
                           num_epochs=epochs,
                           base_lr=learning_rate,
                           final_lr=learning_rate / 100,
                           iter_per_epoch=1)

for epoch in range(epochs):
    loss_lis = []
    l2_loss_lis = []
    # model training
    model.train()
    cosineLR.step()
    for i, input in enumerate(train_loader):
        # adjust learning rate
        # 此处可以换成余弦退火
        # if epoch == 300:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = learning_rate * 0.1
        input = input.cuda()
        # compute output
        output = model(input)
        # loss = criterion(output * 128, input * 128)
        loss = criterion(output * 128, input * 128)
        l2_loss = l2_regularization_loss(model=model, l2_alpha=l2_alpha)*cfg.LAMDA
        mixed_loss = l2_loss + loss
        # compute gradient and do Adam step
        optimizer.zero_grad()
        # loss.backward()
        # mixed_loss.backward(retain_graph=True)
        l2_loss.backward(retain_graph=True)
        # l2_regularization(model=model, l2_alpha=l2_alpha)
        optimizer.step()

        # 用于计算后面的loss平均
        loss_lis.append(loss.item())
        l2_loss_lis.append(l2_loss.item())


        if i % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t\tLearning Rate: {3}\tLoss {loss:.6f}\tL2Loss {l2_loss:.6f}\t'
                .format(epoch,
                        i,
                        len(train_loader),
                        cosineLR.get_lr(),
                        loss=loss.item(),
                        l2_loss=l2_loss.item()))
    # 计算loss平均
    ave_loss = sum(loss_lis) / len(loss_lis)
    l2_ave_loss = sum(l2_loss_lis) / len(l2_loss_lis)
    # 画图
    writer.add_scalar('MSE_LOSS', ave_loss, epoch + 1)
    writer.add_scalar('L2_LOSS', l2_ave_loss, epoch + 1)
    writer.add_scalar('MIXED_LOSS', (l2_ave_loss+ave_loss)/2, epoch + 1)

    # model evaluating
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            total_loss += criterion(output, input).item() * input.size(0)
        average_loss = total_loss / len(test_dataset)
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = '{}/{}/encoder.pth.tar'.format(
                cfg.PROJECT_ROOT, cfg.MODEL_SAVE_PATH)
            torch.save({
                'state_dict': model.encoder.state_dict(),
            }, modelSave1)
            # save decoder
            modelSave2 = '{}/{}/decoder.pth.tar'.format(
                cfg.PROJECT_ROOT, cfg.MODEL_SAVE_PATH)
            torch.save({
                'state_dict': model.decoder.state_dict(),
            }, modelSave2)
            print("Model saved")
            best_loss = average_loss
