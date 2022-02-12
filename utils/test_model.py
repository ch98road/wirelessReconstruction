import torch
import os
from utils.LoadWeight import LoadWeight, InitWeight


def test_model(model,
               in_shape=(1, 2, 126, 128),
               pretrain=False,
               path=None,
               excepts=[],
               devices='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    # InitWeight(model)
    # model.cuda()
    input = torch.rand(in_shape).cuda()
    # inp = input.view(1, 32, 32, 32)
    if pretrain and path is not None:
        LoadWeight(model, path, excepts)
    model(input)
    print('finished')



