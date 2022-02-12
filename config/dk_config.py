import os

PROJECT_ROOT = '/work/workspace3/chenhuan/code/wirelessReconstruction'
CUDA_VISIBLE_DEVICES = '3'
# MODEL_SAVE_PATH = 'Modelsave_AttDecoder'
# MODEL_SAVE_PATH = 'Modelsave_DepthWise'
version_name = 'dk_model'
MODEL_SAVE_PATH = 'Modelsave/{}'.format(version_name)
LOG_PATH = 'dk_model'

# PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'encoder.pth.tar')
# PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'decoder.pth.tar')
PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/{}'.format(version_name),
                                'encoder.pth.tar')
PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/{}'.format(version_name),
                                'decoder.pth.tar')

SAVE = True


use_single_gpu = True  # select whether using single gpu or multiple gpus
batch_size = 256
epochs = 10000
learning_rate = 1e-5
num_workers = 4
print_freq = 60  # print frequency (default: 60)
# parameters for data
feedback_bits = 512
PRETRAIN = True
# 是否采用L2
L2LOSS = True
l2_alpha = 0.3
LAMDA = 0.2
