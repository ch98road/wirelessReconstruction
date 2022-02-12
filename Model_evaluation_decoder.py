#!/usr/bin/env python3
import numpy as np
import h5py
# from Model_define_pytorch import NMSE, AutoEncoder, DatasetFolder
from model.Model_define_pytorch_officeWithCA import AutoEncoder, DatasetFolder, NMSE

import torch
import os
import config.config as cfg


def Score(NMSE):
    score = (1 - NMSE) * 100
    return score


def test():
    # test
    model_decoder.eval()
    y_test = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model_decoder(input)
            output = output.cpu().numpy()
            if i == 0:
                y_test = output
            else:
                y_test = np.concatenate((y_test, output), axis=0)
    return y_test


if __name__ == '__main__':
    # Parameters for training
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    batch_size = 64
    num_workers = 4
    # parameter setting

    feedback_bits = 512
    # Data loading
    import scipy.io as scio
    # load test data
    data_load_address = '{}/data'.format(cfg.PROJECT_ROOT)
    mat = scio.loadmat(data_load_address + '/Htest.mat')
    x_test = mat['H_test']  # shape=？*126*128*2

    x_test = np.transpose(x_test.astype('float32'), [0, 3, 1, 2])

    # load encoder_output
    decode_input = np.load('{}/{}/encoder_output.npy'.format(
        cfg.PROJECT_ROOT, cfg.MODEL_SAVE_PATH))

    # load model and test NMSE
    model = AutoEncoder(feedback_bits).cuda()
    model_decoder = model.decoder
    model_path = '{}/{}/decoder.pth.tar'.format(cfg.PROJECT_ROOT,
                                                cfg.MODEL_SAVE_PATH)
    model_decoder.load_state_dict(torch.load(model_path)['state_dict'])
    print("weight loaded")

    # dataLoader for test
    test_dataset = DatasetFolder(decode_input)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True)

    y_test = test()

    # need convert channel first to channel last for evaluate.
    print('The NMSE is ' + np.str(
        NMSE(np.transpose(x_test, (0, 2, 3,
                                   1)), np.transpose(y_test, (0, 2, 3, 1)))))

    NMSE_test = NMSE(np.transpose(x_test, (0, 2, 3, 1)),
                     np.transpose(y_test, (0, 2, 3, 1)))
    scr = Score(NMSE_test)
    if scr < 0:
        scr = 0
    else:
        scr = scr

    result = 'score=', np.str(scr)
    print(result)