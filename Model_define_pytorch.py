#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
from json import encoder
from turtle import forward
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math


# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1, ))
        out = integer.unsqueeze(-1) // 2**exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2**(B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2**B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant),
                             dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2**B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv1x1(in_planes=in_planes,
                             out_planes=mid_planes,
                             stride=stride)
        self.conv2 = conv3x3(in_planes=mid_planes,
                             out_planes=mid_planes,
                             stride=stride)
        self.conv3 = conv1x1(in_planes=mid_planes,
                             out_planes=in_planes,
                             stride=stride)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        return out + x


class ResBlockWithCBAM(nn.Module):
    def __init__(self, in_planes=2, mid_planes=None, stride=1):
        super(ResBlockWithCBAM, self).__init__()
        if mid_planes is None:
            mid_planes = in_planes * 4
        self.conv1 = DepthWiseConv(in_ch=in_planes,out_ch=mid_planes)
        self.conv2 = DepthWiseConv(in_ch=mid_planes,out_ch=mid_planes)
        self.conv3 = DepthWiseConv(in_ch=mid_planes,out_ch=in_planes)
                             
        self.channelAtt = ChannelAttention(in_planes=in_planes)
        self.spatialAtt = SpatialAttention()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.channelAtt(out) * out
        out = self.spatialAtt(out) * out
        return out + x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthWiseConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2,
                               1,
                               kernel_size,
                               padding=kernel_size // 2,
                               bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.first_conv = DepthWiseConv(2, 2)
        self.conv1 = DepthWiseConv(2, 16)
        self.res1 = ResBlockWithCBAM(16, 16)
        self.conv2 = DepthWiseConv(16, 32)
        self.res2 = ResBlockWithCBAM(32, 32)
        self.res3 = ResBlockWithCBAM(32, 32)
        self.conv3 = DepthWiseConv(32, 2)
        self.fc = nn.Linear(32256, int(feedback_bits // self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        first_out = F.relu(self.first_conv(x))

        out = F.relu(self.conv1(first_out))
        res = self.res1(out) + out
        out = F.relu(self.conv2(res))
        res = self.res2(out) + out
        res = self.res3(res) + res + out
        out = F.relu(self.conv3(res)) + first_out

        out = out.view(-1, 32256)
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)

        return out


class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()

        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)

        self.layers = [[2, 16], [16, 64], [64, 16]]

        self.multiConvs = nn.ModuleList()
        self.after_conv = DepthWiseConv(in_ch=self.layers[-1][1], out_ch=2)

        self.fc = nn.Linear(int(feedback_bits // self.B), 32256)
        self.out_cov = DepthWiseConv(2, 2)
        self.sig = nn.Sigmoid()

        for layer in self.layers:
            self.multiConvs.append(
                nn.Sequential(conv1x1(layer[0], layer[1]),
                              ResBlockWithCBAM(in_planes=layer[1])))

    def forward(self, x):
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits // self.B))
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 126, 128)
        # for i in range(3):
        #     residual = out
        #     out = self.multiConvs[i](out)
        #     out = residual + out

        for index in range(len(self.layers)):
            out = self.multiConvs[index](out)
        out = F.relu(self.after_conv(out))
        out = self.out_cov(out)
        out = self.sig(out)
        return out


# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         self.multiConvs = nn.ModuleList()
#         self.fc = nn.Linear(int(feedback_bits // self.B), 32256)
#         self.out_cov = conv3x3(2, 2)
#         self.sig = nn.Sigmoid()

#         for _ in range(3):
#             self.multiConvs.append(
#                 nn.Sequential(conv3x3(2, 8), nn.ReLU(), conv3x3(8, 16),
#                               nn.ReLU(), conv3x3(16, 2), nn.ReLU()))

#     def forward(self, x):
#         out = self.dequantize(x)
#         out = out.view(-1, int(self.feedback_bits // self.B))
#         out = self.sig(self.fc(out))
#         out = out.view(-1, 2, 126, 128)
#         for i in range(3):
#             residual = out
#             out = self.multiConvs[i](out)
#             out = residual + out

#         out = self.out_cov(out)
#         out = self.sig(out)
#         return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C)**2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C)**2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]


if __name__ == '__main__':
    encoder = Encoder(feedback_bits=512)