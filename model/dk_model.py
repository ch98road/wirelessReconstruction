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
    # num = torch.zeros(Bit_[:, :, 1].shape)
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
def conv5x5(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(kernel_size=(5, 5), stride=(2, 2), out_channels=32, in_channels=1)

class ResBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, stride=1):
        super(ResBlock,self).__init__()
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=mid_planes, stride=stride)
        self.conv2 = conv3x3(in_planes=mid_planes, out_planes=mid_planes, stride=stride)
        self.conv3 = conv1x1(in_planes=mid_planes, out_planes=in_planes, stride=stride)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        return out+x



class ResBlockWithCBAM(nn.Module):
    def __init__(self, in_planes, mid_planes, stride=1):
        super(ResBlockWithCBAM,self).__init__()
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=mid_planes, stride=stride)
        self.conv2 = conv3x3(in_planes=mid_planes, out_planes=mid_planes, stride=stride)
        self.conv3 = conv1x1(in_planes=mid_planes, out_planes=in_planes, stride=stride)
        self.channelAtt = ChannelAttention(in_planes=in_planes)
        self.spatialAtt = SpatialAttention()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.channelAtt(out)*out
        out = self.spatialAtt(out)*out
        return out+x


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
        self.zero_padding = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
        )  # B*2*128*128
        self.conv1 = conv3x3(2, 256) #B*C*128*128
        self.conv_downsample1 = nn.Sequential(
            nn.Conv2d(kernel_size=(4, 4), stride=(2, 2), out_channels=128, in_channels=256, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # B*C*64*64
        self.conv_downsample2 = nn.Sequential(
            nn.Conv2d(kernel_size=(4, 4), stride=(2, 2), out_channels=64, in_channels=128, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # B*C*32*32
        self.res1 = ResBlockWithCBAM(64,64)
        self.conv_downsample3 = nn.Sequential(
            nn.Conv2d(kernel_size=(4, 4), stride=(2, 2), out_channels=32, in_channels=64, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # B*C*16*16
        self.res2 = ResBlockWithCBAM(32, 32)
        self.conv_downsample4 = nn.Sequential(
            nn.Conv2d(kernel_size=(4, 4), stride=(2, 2), out_channels=16, in_channels=32, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # B*C*8*8

        self.conv2 = conv3x3(16, 2) #B*C*128*128
        # encoder 128—> 2*8*8
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        # print("x0", x.shape)
        # 128*126*2 -》padding 128*128*2
        x = self.zero_padding(x)
        # print("x1", x.shape)
        out = F.relu(self.conv1(x))
        out = self.conv_downsample1(out)
        out = self.conv_downsample2(out)
        out = self.res1(out)
        out = self.conv_downsample3(out)
        out = self.res2(out)
        out = self.conv_downsample4(out)
        out = self.conv2(out)
        out = out.view(-1, 128)
        out = self.sig(out)
        out = self.quantize(out)
        return out


class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)

        self.upsample_block1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),  # 16*16
            nn.Conv2d(kernel_size=(3, 3), stride=(1, 1), out_channels=512, in_channels=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.upsample_block2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),  # 32*32
            nn.Conv2d(kernel_size=(3, 3), stride=(1, 1), out_channels=256, in_channels=512, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample_block3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),  # 64*64
            nn.Conv2d(kernel_size=(3, 3), stride=(1, 1), out_channels=128, in_channels=256, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample_block4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),  # 128*128
            nn.Conv2d(kernel_size=(3, 3), stride=(1, 1), out_channels=64, in_channels=128, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_channel = conv3x3(64, 2)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.dequantize(x)
        # print("dequan",out.shape)
        # out = out.view(-1, int(self.feedback_bits // self.B))
        # print(out.shape)
        out = out.view(-1,2, 8,8)
        # print(out.shape)
        out = self.upsample_block1(out)
        # print("upsample",out.shape)
        out = self.upsample_block2(out)
        out = self.upsample_block3(out)
        out = self.upsample_block4(out)
        out = self.conv_channel(out)
        # print(out.shape)
        out = self.sig(out)
        out = out[:, :, 1:-1,]
        return out


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
    print(encoder.shape)