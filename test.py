# # 测试self-att的性能
# import torch

# from model.Model_define_pytorch_SelfAtt import BottleneckAttention

# att = BottleneckAttention(dim=32, fmap_size=(32, 32), heads=4).cuda()
# inp = torch.rand(32, 32, 32, 32).cuda()
# print(inp.shape)
# # inp.view(-1, 8, 31, 32)
# print(inp.shape)

# y = att(inp)
# assert y.shape == inp.shape
# print(y.shape)

# # 加载文件测试
# from codecs import EncodedFile
# import torch
# import config.config_selfAtt as cfg
# from model.Model_define_pytorch_SelfAtt import AutoEncoder, DatasetFolder, Encoder
# from utils.LoadWeight import LoadWeight

# feedback_bits = 512
# model = Encoder(feedback_bits)
# # model_dict = model.state_dict()
# pretrained_dict = torch.load(cfg.PRE_ENCODER_PATH)['state_dict']
# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # model_dict.update(pretrained_dict)
# # model.load_state_dict(model_dict)
# LoadWeight(model, cfg.PRE_ENCODER_PATH,excepts=['fc'])
# pass

# # 数据padding测试
# import numpy as np
# import torch

# a = np.array([3, 4, 5])
# b = np.array([0, 0, 0])

# t1 = torch.from_numpy(a)
# t2 = torch.from_numpy(b)
# print('t1:', t1)
# print('------------------------')
# print('t2:', t2)

# res = torch.cat((t1, t2), 0)
# print(res)

# inp = torch.rand(32, 8, 126, 128)
# inp2 = torch.zeros(32, 8, 2, 128)
# res = torch.cat((inp, inp2), 2)
# print(res.shape)
# print(inp.shape)

# import scipy.io as scio
# data_load_address = '{}/data'.format(cfg.PROJECT_ROOT)
# mat = scio.loadmat(data_load_address + '/Htest.mat')
# x_test = mat['H_test']  # shape=2000*126*128*2

# import torch
# import os
# from model.Model_define_pytorch_SelfAtt import AutoEncoder, DatasetFolder, Encoder
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# model = Encoder(512).cuda()
# input = torch.rand(256, 2, 126, 128).cuda()
# # inp = input.view(1, 32, 32, 32)
# model(input)
# print('finished')
# pass


# test Model
# import torch
# import os
# from model.Model_define_pytorch_SAwithCACRB_B_Quant import AutoEncoder, DatasetFolder, Encoder
# from utils.test_model import *
# import config.config_SACADecoder as cfg
# model = AutoEncoder(512).cuda()
# model.encoder.quantization = False
# model.decoder.quantization = False
# test_model(model,
#            in_shape=(64, 2, 126, 128),
#            pretrain=False,
#            path=None)

# test TrainLoader
# import torch
# from utils.DataLoader import DataLoader
# torch.distributed.init_process_group(backend="nccl")

# data, dataset, loader = DataLoader(distributed=True)
# pass


# test FomatSave
# from ast import For
# from model.Model_define_pytorch_CRBCA_LQ import AutoEncoder
# from utils.utils import FormatSaveAutoEncoder
# model = AutoEncoder(512)
# savePath = '/work/workspace3/chenhuan/code/wirelessReconstruction/Modelsave/CRBCAv1LQv1'

# FormatSaveAutoEncoder(model=model, savePath=savePath)


# test mode eval
import torch
import os
# from model.Model_define_pytorch_CRBCA_LQ import AutoEncoder
from model.Model_define_pytorch_office import AutoEncoder, DatasetFolder, NMSE

from utils.model_eval import eval
model = AutoEncoder(512)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
savePath = '/work/workspace3/chenhuan/code/wirelessReconstruction/Modelsave/BaseLineMulti'
dataPath = '/work/workspace3/chenhuan/code/wirelessReconstruction/data/Htest.mat'
checkpoint_PATH = os.path.join(savePath, 'AutoEncoder.pth.tar')
model_CKPT = torch.load(checkpoint_PATH)
model.load_state_dict(model_CKPT['state_dict'])
model = model.cuda()

print('loading checkpoint!')
eval(model, data_oath=dataPath)

# import torch
# import os
# from utils.DataLoader import ReadData
# import numpy as np


# test_data = ReadData(
#     datapath='/work/workspace3/chenhuan/code/wirelessReconstruction/data/Htest.mat', datatype='test')

# data = ReadData()
# cat_data = np.concatenate([test_data, data], axis=0)
# mean = np.mean(cat_data)
# std = np.std(cat_data)
# new_data = (cat_data-mean)/std
# ret_data = new_data*std+mean
# pass
# mean = 0.50001776
# std = 0.014204533
