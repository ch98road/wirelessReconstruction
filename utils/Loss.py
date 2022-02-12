import torch
from torch import nn
from torch.autograd import Variable


def l1_regularization_loss(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization_loss(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight**2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.weight.grad.data.add_(l1_alpha *
                                         torch.sign(module.weight.data))


def l2_regularization(model, l2_alpha):
    for module in model.modules():
        if type(module) is nn.Conv2d:
            module.weight.grad.data.add_(l2_alpha * module.weight.data)


# def NMSE_LOSS(output, input):
#     in_real = input[:, 0].view(-1, input.shape[2] * input.shape[3])
#     in_imag = input[:, 1].view(-1, input.shape[2] * input.shape[3])
#     out_real = output[:, 0].view(-1, output.shape[2] * output.shape[3])
#     out_imag = output[:, 1].view(-1, output.shape[2] * output.shape[3])
#     in_C = in_real - 0.5 + 1j * (in_imag - 0.5)
#     out_C = out_real - 0.5 + 1j * (out_imag - 0.5)
#     power = torch.sum(torch.abs(out_C)**2, dim=1)
#     mse = torch.sum(torch.abs(out_C - in_C)**2, dim=1)
#     nmse = torch.mean(mse / power)
#     return nmse


def NMSE_POWER(output):
    out_real = output[:, 0].view(-1, output.shape[2] * output.shape[3])
    out_imag = output[:, 1].view(-1, output.shape[2] * output.shape[3])
    out_C = out_real - 0.5 + 1j * (out_imag - 0.5)
    power = torch.mean(torch.sum(torch.abs(out_C)**2, dim=1), dim=0)
    return power


def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real - x_hat_real)**2 + (x_imag - x_hat_imag)**2,
                    axis=1)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse
