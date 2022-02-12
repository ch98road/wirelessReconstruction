import torch
import numpy as np
from utils.DataLoader import DataLoader
from utils.Loss import NMSELoss
from utils.utils import TestOneEpoch, Score
from Model_define_pytorch import NMSE


def eval(model, data_oath, batch_size=8, num_workers=4):
    x_test, test_dataset, test_loader = DataLoader(datapath=data_oath,
                                                   datatype='test',
                                                   batch_size=batch_size,
                                                   num_workers=4)
    nmseloss = NMSELoss(reduction='mean').cuda()
    # model evaluating
    test_loss, y_test = TestOneEpoch(model=model,
                                     test_loader=test_loader,
                                     criterion=nmseloss,
                                     len_test=len(test_dataset))
    NMSE_test = NMSE(np.transpose(x_test, (0, 2, 3, 1)),
                     np.transpose(y_test, (0, 2, 3, 1)))
    score = Score(NMSE_test)
    print('The NMSE is ' + str(NMSE_test))
    print('score={}'.format(score))
    print('testMSE={}'.format(test_loss))
