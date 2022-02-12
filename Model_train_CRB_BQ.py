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
# 这个版本Quant策略不同
# import h5py
import os
import torch
import numpy as np
import torch.nn as nn
import scipy.io as scio
import config.config_CRB_BQ as cfg
from tensorboardX import SummaryWriter
import utils.CosineAnnealingWithWarmup as LR
from model.Model_define_pytorch_CRB_BQ import AutoEncoder, DatasetFolder, NMSE
from model.adversarial import Adversarial
from utils.LoadWeight import LoadWeight,InitWeight
from utils.Loss import l2_regularization, l2_regularization_loss, NMSE_POWER, NMSELoss
from utils.DataLoader import DataLoader
from utils.utils import LogScalar, SaveAutoEncoder, TestOneEpoch, TrainOneEpoch, Score

if not os.path.exists('{}/log/{}'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH)):
    os.mkdir('{}/log/{}'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))

writer = SummaryWriter('{}/log/{}/'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
torch.manual_seed(1)


# Model construction
model = AutoEncoder(cfg.feedback_bits)
# initialization
InitWeight(model)
if cfg.PRETRAIN:
    # model.encoder.load_state_dict(torch.load(cfg.PRE_ENCODER_PATH)['state_dict'])
    # model.decoder.load_state_dict(torch.load(cfg.PRE_DECODER_PATH)['state_dict'])
    # LoadWeight(model.encoder, cfg.PRE_ENCODER_PATH)
    LoadWeight(model.decoder, cfg.PRE_DECODER_PATH)
    print("weight loaded")

if cfg.use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

# criterion = nn.MSELoss().cuda()
nmseloss = NMSELoss(reduction='sum').cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Data loading
data_load_address = '{}/data'.format(cfg.PROJECT_ROOT)
x_train, _, train_loader = DataLoader(datapath=data_load_address +
                                      '/Htrain.mat',
                                      datatype='train',
                                      batch_size=cfg.batch_size,
                                      num_workers=cfg.num_workers)
x_test, test_dataset, test_loader = DataLoader(datapath=data_load_address +
                                               '/Htest.mat',
                                               datatype='test',
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers)

best_loss = 256
cosineLR = LR.LR_Scheduler(optimizer=optimizer,
                           warmup_epochs=5,
                           warmup_lr=cfg.learning_rate,
                           num_epochs=cfg.epochs,
                           base_lr=cfg.learning_rate,
                           final_lr=cfg.learning_rate / 100,
                           iter_per_epoch=1)

for epoch in range(cfg.epochs):
    # model training
    cosineLR.step()
    if epoch < cfg.epochs // 10:
        try:
            model.encoder.quantization = False
            model.decoder.quantization = False
        except:
            model.module.encoder.quantization = False
            model.module.decoder.quantization = False
    else:
        try:
            model.encoder.quantization = True
            model.decoder.quantization = True
        except:
            model.module.encoder.quantization = True
            model.module.decoder.quantization = True
    loss_lis, l2_loss_lis = TrainOneEpoch(model=model,
                                          train_loader=train_loader,
                                          criterion=nmseloss,
                                          L2=cfg.L2LOSS,
                                          L2_ALPHA=cfg.l2_alpha,
                                          optimizer=optimizer)

    # 计算loss平均
    ave_loss = sum(loss_lis) / len(loss_lis)
    if cfg.L2LOSS:
        l2_ave_loss = sum(l2_loss_lis) / len(l2_loss_lis)
    else:
        l2_ave_loss = 0
    print('[{}/{}]\tLOSS:{:.6f}\tL2_LOSS:{:.6f}\tLR:{:.6f}\t'.format(epoch,
          cfg.epochs, ave_loss-l2_ave_loss, l2_ave_loss, cosineLR.get_lr()))
    # model evaluating
    test_loss, y_test = TestOneEpoch(model=model,
                                     test_loader=test_loader,
                                     criterion=nmseloss,
                                     len_test=len(test_dataset))
    if test_loss < best_loss and cfg.SAVE:
        savePath = os.path.join(cfg.PROJECT_ROOT, cfg.MODEL_SAVE_PATH)
        # model save
        SaveAutoEncoder(model=model, savePath=savePath)
        best_loss = test_loss
    NMSE_test = NMSE(np.transpose(x_test, (0, 2, 3, 1)),
                     np.transpose(y_test, (0, 2, 3, 1)))
    score = Score(NMSE_test)
    print('The NMSE is ' + str(NMSE_test))
    print('score={}'.format(score))
    print('testMSE={}'.format(test_loss))

    # 画图
    LogScalar(writer=writer,
              epoch=epoch,
              ave_loss=ave_loss,
              l2_ave_loss=None if l2_ave_loss == 0 else l2_ave_loss,
              NMSE_test=NMSE_test,
              score=score,
              test_loss=test_loss)
