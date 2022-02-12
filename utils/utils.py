import torch.nn as nn
import torch
from utils.Loss import l2_regularization_loss
import numpy as np
import os


def TrainOneEpoch(model=None,
                  train_loader=None,
                  criterion=nn.MSELoss().cuda(),
                  L2=True,
                  L2_ALPHA=0.1,
                  optimizer=None,
                  print_freq=100):
    assert model is not None, 'model is None'
    assert optimizer is not None, 'optimizer is None'
    assert train_loader is not None, 'train_loader is None'

    loss_lis = []
    l2_loss_lis = []
    # model training
    model.train()

    for i, input in enumerate(train_loader):
        input = input.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, input)
        if L2:
            l2_loss = l2_regularization_loss(model=model, l2_alpha=L2_ALPHA)
            l2_loss_lis.append(l2_loss.item())
            loss += l2_loss

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 用于计算后面的loss平均
        # loss_lis.append(loss.item())
        loss_lis.append(loss.item())
        if i % print_freq == 0:
            print("iter = {}:loss = {}".format(i, loss))
            # print("insize:{},outsize:{}".format(input.shape, output.shape))

    return loss_lis, l2_loss_lis


def TestOneEpoch(
        model=None,
        test_loader=None,
        criterion=nn.MSELoss().cuda(),
        len_test=2000,
):
    assert model is not None, 'model is None'
    assert test_loader is not None, 'test_loader is None'
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
        average_loss = total_loss / len_test

    return average_loss, y_test


def SaveAutoEncoderDistributed(model=None, savePath=None):
    assert model is not None, 'model is None'
    assert savePath is not None, 'savePath is None'
    try:
        if not os.path.exists(savePath):
            os.mkdir(savePath)
    except:
        print("Cannot mkdir savePath, please check!")
    modelSave = os.path.join(savePath, 'AutoEncoder.pth.tar')
    torch.save({
        'state_dict': model.module.state_dict(),
    }, modelSave)


def SaveAutoEncoder(model=None, savePath=None):
    assert model is not None, 'model is None'
    assert savePath is not None, 'savePath is None'
    try:
        if not os.path.exists(savePath):
            os.mkdir(savePath)
    except:
        print("Cannot mkdir savePath, please check!")

    # save encoder
    modelSave1 = os.path.join(savePath, 'encoder.pth.tar')

    torch.save({
        'state_dict': model.encoder.state_dict(),
    }, modelSave1)
    # save decoder
    modelSave2 = os.path.join(savePath, 'decoder.pth.tar')

    torch.save({
        'state_dict': model.decoder.state_dict(),
    }, modelSave2)
    print("Model saved")


def FormatSaveAutoEncoder(model=None, savePath=None):
    assert model is not None, 'model is None'
    assert savePath is not None, 'savePath is None'
    try:
        if not os.path.exists(savePath):
            os.mkdir(savePath)
    except:
        print("Cannot mkdir savePath, please check!")

    checkpoint_PATH = os.path.join(savePath, 'AutoEncoder.pth.tar')
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])

    print('loading checkpoint!')
    # save encoder
    modelSave1 = os.path.join(savePath, 'encoder.pth.tar')

    torch.save({
        'state_dict': model.encoder.state_dict(),
    }, modelSave1)
    # save decoder
    modelSave2 = os.path.join(savePath, 'decoder.pth.tar')

    torch.save({
        'state_dict': model.decoder.state_dict(),
    }, modelSave2)
    print("Model saved")


def LogScalar(writer=None,
              epoch=0,
              ave_loss=None,
              l2_ave_loss=None,
              NMSE_test=None,
              score=None,
              test_loss=None):
    pass
    assert writer is not None

    if ave_loss is not None:
        writer.add_scalar('MSE_LOSS', ave_loss, epoch + 1)
    if l2_ave_loss is not None:
        writer.add_scalar('L2_LOSS', l2_ave_loss, epoch + 1)
    if NMSE_test is not None:
        writer.add_scalar('NMSE', NMSE_test, epoch + 1)
    if score is not None:
        writer.add_scalar('score', score, epoch + 1)
    if test_loss is not None:
        writer.add_scalar('testMSE', test_loss, epoch + 1)


def Score(NMSE):
    score = (1 - NMSE) * 100
    return score
