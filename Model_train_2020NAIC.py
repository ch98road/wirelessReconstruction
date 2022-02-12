'''
seefun . Aug 2020.
Modified by Naive . Sep 2020.
1. add L1-norm regularization on the NMSE loss
2. adjust the number of kernel
3. use different initilazations
4. learning rate adjustment according to reference
'''

import numpy as np
import h5py
import torch
import os
import math
import torch.nn as nn
import random
from torch.nn import init
import scipy.io as scio

from model.Model_define_pytorch_2020 import AutoEncoder, DatasetFolder, NMSE_cuda, NMSELoss, NMSE
import config.config_2020 as cfg
from utils.LoadWeight import LoadWeight
from utils.DataLoader import DataLoader
from utils.utils import *
from tensorboardX import SummaryWriter
writer = SummaryWriter('{}/log/{}/'.format(cfg.PROJECT_ROOT, cfg.LOG_PATH))


def Score(NMSE):
    score = (1 - NMSE) * 100
    return score


# Parameters for training
gpu_list = cfg.CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def seed_everything(seed=258):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(cfg.SEED)


# Model construction
model = AutoEncoder(cfg.feedback_bits)

model.encoder.quantization = False
model.decoder.quantization = False

# initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform(m.weight,
                                mode='fan_in',
                                nonlinearity='leaky_relu')

if cfg.PRETRAIN:
    # model.encoder.load_state_dict(torch.load(cfg.PRE_ENCODER_PATH)['state_dict'])
    # model.decoder.load_state_dict(torch.load(cfg.PRE_DECODER_PATH)['state_dict'])
    LoadWeight(model.encoder, cfg.PRE_ENCODER_PATH)
    LoadWeight(model.decoder, cfg.PRE_DECODER_PATH)
    print("weight loaded")
if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()

criterion = NMSELoss(reduction='mean')  # nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

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
# best_loss = 1

best_loss = 1
for epoch in range(cfg.epochs):
    loss_lis = []
    print('========================')
    print('lr:%.4e' % optimizer.param_groups[0]['lr'])
    # model training
    model.train()
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

    # lr modified according to cosine
    optimizer.param_groups[0]['lr'] = cfg.learning_rate_final + 0.5 * (
        cfg.learning_rate_init - cfg.learning_rate_final) * (1 + math.cos(
            (epoch * 3.14) / cfg.epochs))

    loss_lis, l2_loss_lis = TrainOneEpoch(model=model,
                                          train_loader=train_loader,
                                          criterion=criterion,
                                          L2=True,
                                          L2_ALPHA=0.03,
                                          optimizer=optimizer)
    # 画loss
    ave_loss = sum(loss_lis) / len(loss_lis)
    writer.add_scalar('NMSE_LOSS', ave_loss, epoch + 1)

    model.eval()
    try:
        model.encoder.quantization = True
        model.decoder.quantization = True
    except:
        model.module.encoder.quantization = True
        model.module.decoder.quantization = True
    ave_loss = sum(loss_lis) / len(loss_lis)
    l2_ave_loss = sum(l2_loss_lis) / len(l2_loss_lis)
    print('[{}/{}]\tLOSS:{:.6f}\tL2_LOSS:{:.6f}\t'.format(epoch,
          cfg.epochs, ave_loss-l2_ave_loss, l2_ave_loss))
    # model evaluating
    test_loss, y_test = TestOneEpoch(model=model,
                                     test_loader=test_loader,
                                     criterion=criterion_test,
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
              l2_ave_loss=l2_ave_loss,
              NMSE_test=NMSE_test,
              score=score,
              test_loss=test_loss)
