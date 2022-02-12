import torch
import torch.nn as nn


def LoadWeight(model, weight_path, excepts=[]):

    model_dict = model.state_dict()
    pretrained_dict = torch.load(weight_path)['state_dict']
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model_dict
        and k.split('.')[0] not in excepts and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def InitWeight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform(m.weight,
                                    mode='fan_in',
                                    nonlinearity='leaky_relu')


