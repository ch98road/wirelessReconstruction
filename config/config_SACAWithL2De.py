import os

PROJECT_ROOT = '/work/workspace3/chenhuan/code/wirelessReconstruction'
CUDA_VISIBLE_DEVICES = '1'
# MODEL_SAVE_PATH = 'Modelsave_AttDecoder'
# MODEL_SAVE_PATH = 'Modelsave_DepthWise'
MODEL_SAVE_PATH = 'Modelsave/SACADecoder2'
LOG_PATH = 'SACADecoder2'

# PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'encoder.pth.tar')
# PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'decoder.pth.tar')
PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/SelfAttWithCA',
                                'encoder.pth.tar')
PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/SelfAttWithCA',
                                'decoder.pth.tar')

SAVE = True


use_single_gpu = True  # select whether using single gpu or multiple gpus
batch_size = 128
epochs = 1500
learning_rate = 1e-3
num_workers = 4
print_freq = 60  # print frequency (default: 60)
# parameters for data
feedback_bits = 512
PRETRAIN = False
l2_alpha = 0.03
LAMDA = 0.2
