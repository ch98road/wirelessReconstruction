import os

PROJECT_ROOT = '/work/workspace3/chenhuan/code/wirelessReconstruction'
CUDA_VISIBLE_DEVICES = '5'
# MODEL_SAVE_PATH = 'Modelsave_AttDecoder'
# MODEL_SAVE_PATH = 'Modelsave_DepthWise'
MODEL_SAVE_PATH = 'Modelsave/OfficeWithPre'
LOG_PATH = 'OfficeWithPre'
LAMDA = 0.2

# PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'encoder.pth.tar')
# PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'decoder.pth.tar')
PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/OfficeWithPre',
                                'encoder.pth.tar')
PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, 'Modelsave/OfficeWithPre',
                                'decoder.pth.tar')