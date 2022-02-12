import os

PROJECT_ROOT = '/work/workspace3/chenhuan/code/wirelessReconstruction'
CUDA_VISIBLE_DEVICES = '2'
# MODEL_SAVE_PATH = 'Modelsave_AttDecoder'
# MODEL_SAVE_PATH = 'Modelsave_DepthWise'
# MODEL_SAVE_PATH = 'Modelsave/OfficeWithCAL2'
MODEL_SAVE_PATH = 'codeDemo/NAIC_pytorch_2021/submissions/project'
LOG_PATH = 'OfficeWithCA_eval'
LAMDA = 0.2

# PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'encoder.pth.tar')
# PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_PATH,
#                                 'decoder.pth.tar')
PRE_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'codeDemo/NAIC_pytorch_2021/submissions/project',
                                'encoder.pth.tar')
PRE_DECODER_PATH = os.path.join(PROJECT_ROOT, 'codeDemo/NAIC_pytorch_2021/submissions/project',
                                'decoder.pth.tar')
resnet_model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}