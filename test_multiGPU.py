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
import config.config_CRBCA_BQ as cfg
# from tensorboardX import SummaryWriter
import utils.CosineAnnealingWithWarmup as LR
from model.Model_define_pytorch_CRBCA_BQ import AutoEncoder, DatasetFolder, NMSE
from model.adversarial import Adversarial
from utils.LoadWeight import LoadWeight, InitWeight
from utils.Loss import l2_regularization, l2_regularization_loss, NMSE_POWER, NMSELoss
from utils.DataLoader import DataLoader
from utils.utils import LogScalar, SaveAutoEncoder, TestOneEpoch, TrainOneEpoch, Score

if not os.path.exists('{}/log/{}'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH)):
    os.mkdir('{}/log/{}'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))

# writer = SummaryWriter('{}/log/{}/'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
torch.manual_seed(1)


# Model construction
model = AutoEncoder(cfg.feedback_bits)
# initialization
# InitWeight(model)
# if cfg.PRETRAIN:
#     # model.encoder.load_state_dict(torch.load(cfg.PRE_ENCODER_PATH)['state_dict'])
#     # model.decoder.load_state_dict(torch.load(cfg.PRE_DECODER_PATH)['state_dict'])
#     LoadWeight(model.encoder, cfg.PRE_ENCODER_PATH)
#     LoadWeight(model.decoder, cfg.PRE_DECODER_PATH)
#     print("weight loaded")

if cfg.use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[
                                                              local_rank],
                                                          output_device=local_rank)


# criterion = nn.MSELoss().cuda()
nmseloss = NMSELoss(reduction='sum').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
x_train, _, train_loader = DataLoader(batch_size=32, distributed=True)
loss_lis, l2_loss_lis = TrainOneEpoch(model=model,
                                      train_loader=train_loader,
                                      criterion=nmseloss,
                                      L2=False,
                                      L2_ALPHA=cfg.l2_alpha,
                                      optimizer=optimizer)

# for _ in range(10):
#     input = torch.rand(32, 2, 126, 128)
#     output = model(input)
# print("insize:{},outsize:{}".format(input.shape, output.shape))
