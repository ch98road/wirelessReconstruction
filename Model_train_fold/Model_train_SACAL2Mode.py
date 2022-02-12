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
from importlib.resources import path
import numpy as np
# import h5py
import torch
from model.Model_define_pytorch_SAwithCA import AutoEncoder, DatasetFolder, NMSE
from model.adversarial import Adversarial
import os
import torch.nn as nn
import config.config_SACAWithL2 as cfg
import utils.CosineAnnealingWithWarmup as LR
import scipy.io as scio
from utils.LoadWeight import LoadWeight
from utils.Loss import l2_regularization, l2_regularization_loss, NMSE_POWER, NMSELoss
from utils.DataLoader import DataLoader


def Score(NMSE):
    score = (1 - NMSE) * 100
    return score


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
batch_size = 128
epochs = 1000
learning_rate = 1e-4
num_workers = 4
print_freq = 60  # print frequency (default: 60)
# parameters for data
feedback_bits = 512
Pretrain = True
l2_alpha = 0.1

# Model construction
model = AutoEncoder(feedback_bits)
if Pretrain:
    # model.encoder.load_state_dict(torch.load(cfg.PRE_ENCODER_PATH)['state_dict'])
    # model.decoder.load_state_dict(torch.load(cfg.PRE_DECODER_PATH)['state_dict'])
    LoadWeight(model.encoder, cfg.PRE_ENCODER_PATH)
    LoadWeight(model.decoder, cfg.PRE_DECODER_PATH)
    print("weight loaded")

if use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss().cuda()
myloss = NMSELoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
data_load_address = '{}/data'.format(cfg.PROJECT_ROOT)

# Data loading
x_train, _, train_loader = DataLoader(datapath=data_load_address +
                                      '/Htrain.mat',
                                      datatype='train',
                                      batch_size=batch_size,
                                      num_workers=num_workers)
x_test, test_dataset, test_loader = DataLoader(datapath=data_load_address +
                                               '/Htest.mat',
                                               datatype='test',
                                               batch_size=batch_size,
                                               num_workers=num_workers)

best_loss = 1
cosineLR = LR.LR_Scheduler(optimizer=optimizer,
                           warmup_epochs=5,
                           warmup_lr=learning_rate,
                           num_epochs=epochs,
                           base_lr=learning_rate,
                           final_lr=learning_rate / 1000,
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
        # loss = criterion(output*16 , input*16)
        loss2 = myloss(output, input)
        l2_loss = l2_regularization_loss(model=model,
                                         l2_alpha=l2_alpha) * cfg.LAMDA
        # mixed_loss = l2_loss + loss
        loss2 += l2_loss
        # compute gradient and do Adam step
        optimizer.zero_grad()
        # loss.backward()
        # mixed_loss.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()

        # 用于计算后面的loss平均
        # loss_lis.append(loss.item())
        loss_lis.append(loss2.item())
        l2_loss_lis.append(l2_loss.item())

        if i % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t\tLearning Rate: {3}\tLoss {loss:.6f}\tL2_Loss {l2_loss:.6f}\t'
                .format(epoch,
                        i,
                        len(train_loader),
                        cosineLR.get_lr(),
                        loss=loss2.item(),
                        l2_loss=l2_loss.item()))
    # 计算loss平均
    ave_loss = sum(loss_lis) / len(loss_lis)
    l2_ave_loss = sum(l2_loss_lis) / len(l2_loss_lis)

    # 画图
    writer.add_scalar('MSE_LOSS', ave_loss, epoch + 1)
    writer.add_scalar('L2_LOSS', l2_ave_loss, epoch + 1)
    # writer.add_scalar('MIXED_LOSS', (l2_ave_loss+ave_loss)/2, epoch + 1)
    # model evaluating
    model.eval()
    total_loss = 0
    y_test = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            total_loss += criterion(output, input).item() * input.size(0)
            if i == 0:
                y_test = output.cpu()
            else:
                y_test = np.concatenate((y_test, output.cpu()), axis=0)
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
        NMSE_test = NMSE(np.transpose(x_test, (0, 2, 3, 1)),
                         np.transpose(y_test, (0, 2, 3, 1)))
        scr = Score(NMSE_test)
        print('The NMSE is ' + str(NMSE_test))
        print('score={}'.format(scr))
        print('testMSE={}'.format(average_loss))
        writer.add_scalar('NMSE', NMSE_test, epoch + 1)
        writer.add_scalar('score', scr, epoch + 1)
        writer.add_scalar('testMSE', average_loss, epoch + 1)
