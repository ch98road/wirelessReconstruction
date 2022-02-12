import os

PROJECT_ROOT = '/work/workspace3/chenhuan/code/wirelessReconstruction'
CUDA_VISIBLE_DEVICES = '0'
# MODEL_SAVE_PATH = 'Modelsave_AttDecoder'
# MODEL_SAVE_PATH = 'Modelsave_DepthWise'
MODEL_SAVE_PATH = 'Modelsave/2020NAIC'
LOG_PATH = '2020NAIC'
LAMDA = 0.2

# PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'encoder.pth.tar')
# PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'decoder.pth.tar')
PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/SelfAttWithCA',
                                'encoder.pth.tar')
PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/SelfAttWithCA',
                                'decoder.pth.tar')


SEED = 258

batch_size = 2
epochs = 1
learning_rate = 2e-3  # bigger to train faster
learning_rate_init = 2e-3
learning_rate_final = 5e-6
num_workers = 4
print_freq = 500
train_test_ratio = 0.8
L1_weight = 5e-5
# parameters for data
feedback_bits = 512
img_height = 126
img_width = 128
img_channels = 2

SAVE = False
PRETRAIN = False