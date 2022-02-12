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
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange
from torch import einsum
from collections import OrderedDict

from Model_define_pytorch import conv1x1


# borrowed from
# https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py#L21
# i will try to reimplement the function
# as soon as i understand how it works
# not clear to me how it works yet
def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
      Input: [bs, heads, length, 2*length - 1]
      Output: [bs, heads, length, length]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def rel_pos_emb_1d(q, rel_emb, shared_heads):
    """
    Same functionality as RelPosEmb1D
    Args:
        q: a 4d tensor of shape [batch, heads, tokens, dim]
        rel_emb: a 2D or 3D tensor
        of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
    """
    if shared_heads:
        emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
    else:
        emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q
            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head**-0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)


class RelPosEmb2D(nn.Module):
    def __init__(self, feat_map_size, dim_head, heads=None):
        """
        Based on Bottleneck transformer paper
        paper: https://arxiv.org/abs/2101.11605 . Figure 4
        Output: qr^T [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q
            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        self.h, self.w = feat_map_size  # height , width
        self.total_tokens = self.h * self.w
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head, heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head, heads)

    def expand_emb(self, r, dim_size):
        # Decompose and unsqueeze dimension
        r = rearrange(r, 'b (h x) i j -> b h x () i j', x=dim_size)
        expand_index = [-1, -1, -1, dim_size, -1,
                        -1]  # -1 indicates no expansion
        r = r.expand(expand_index)
        return rearrange(r, 'b h x1 x2 y1 y2 -> b h (x1 y1) (x2 y2)')

    def forward(self, q):
        """
        Args:
            q: [batch, heads, tokens, dim_head]
        Returns: [ batch, heads, tokens, tokens]
        """
        assert self.total_tokens == q.shape[
            2], f'Tokens {q.shape[2]} of q must \
        be equal to the product of the feat map size {self.total_tokens} '

        # out: [batch head*w h h]
        r_h = self.emb_w(
            rearrange(q, 'b h (x y) d -> b (h x) y d', x=self.h, y=self.w))
        r_w = self.emb_h(
            rearrange(q, 'b h (x y) d -> b (h y) x d', x=self.h, y=self.w))
        q_r = self.expand_emb(r_h, self.h) + self.expand_emb(r_w, self.h)
        return q_r  # q_r transpose in figure 4 of the paper


class BottleneckAttention(nn.Module):
    def __init__(self,
                 dim,
                 fmap_size,
                 heads=4,
                 dim_head=None,
                 content_positional_embedding=True):
        """
        tensorflow code gist
        https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
        lucidrains' code that I also studied until I finally understood how it works:
        https://github.com/lucidrains/bottleneck-transformer-pytorch
        paper: https://arxiv.org/abs/2101.11605
        Args:
            dim: dimension of the token vector (d_model)
            fmap_size: tuple with the feat map spatial dims
            heads: number of heads representations
            dim_head: inner dimension of the head. dim / heads by default
            content_positional_embedding: whether to include the 2 rel. pos embedding for the query
        """
        super().__init__()
        self.heads = heads
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.scale = self.dim_head**-0.5
        self.fmap_size = fmap_size
        self.content_positional_embedding = content_positional_embedding

        self.to_qkv = nn.Conv2d(dim, heads * self.dim_head * 3, 1, bias=False)

        self.height = self.fmap_size[0]
        self.width = self.fmap_size[1]

        if self.content_positional_embedding:
            self.pos_emb2D = RelPosEmb2D(feat_map_size=fmap_size,
                                         dim_head=self.dim_head)

    def forward(self, x):
        assert x.dim() == 4, f'Expected 4D tensor, got {x.dim()}D tensor'

        # [batch (heads*3*dim_head) height width]
        qkv = self.to_qkv(x)
        # decompose heads and merge spatial dims as tokens
        q, k, v = tuple(
            rearrange(qkv,
                      'b (d k h ) x y  -> k b h (x y) d',
                      k=3,
                      h=self.heads))

        # i, j refer to tokens
        dot_prod = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.content_positional_embedding:
            dot_prod = dot_prod + self.pos_emb2D(q)

        attention = torch.softmax(dot_prod, dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        # Merge heads and decompose tokens to spatial dims
        out = rearrange(out,
                        'b h (x y) d -> b (h d) x y',
                        x=self.height,
                        y=self.width)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


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


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self, in_planes=16, mid_planes=128):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(in_planes, mid_planes, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(mid_planes, mid_planes, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(mid_planes, mid_planes, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(in_planes, mid_planes, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(mid_planes, mid_planes, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(mid_planes * 2, in_planes, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.padding = None
        self.conv1 = conv3x3(2, 2)
        self.conv2 = conv3x3(2, 2)

        self.sa_conv1 = conv3x3(2, 8)
        self.sa_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sa_conv2 = conv3x3(8, 32)
        self.sa_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.att = BottleneckAttention(dim=32, fmap_size=(32, 32), heads=4)

        self.ca_conv1 = conv3x3(2, 2)
        self.ca_conv2 = conv3x3(2, 2)
        self.ca = ChannelAttention(in_planes=2, ratio=1)

        self.fc = nn.Linear(32768, int(feedback_bits // self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        if self.padding is None or self.padding.shape[0] != x.shape[0]:
            self.padding = torch.zeros(x.shape[0], 2, 2, 128).cuda()
        x = torch.cat((x, self.padding), 2)
        # conv
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))

        sa_conv1_out = F.relu(self.sa_conv1(conv2_out))
        sa_avgpool1_out = self.sa_avgpool1(sa_conv1_out)
        sa_conv2_out = F.relu(self.sa_conv2(sa_avgpool1_out))
        sa_avgpool2_out = self.sa_avgpool2(sa_conv2_out)

        ca_conv1_out = F.relu(self.ca_conv1(conv2_out))
        ca_conv2_out = F.relu(self.ca_conv2(ca_conv1_out))
        ca_out = self.ca(ca_conv1_out)*ca_conv2_out

        # selfatt
        assert sa_avgpool2_out.shape[1] == 32 and sa_avgpool2_out.shape[2] == 32
        att_out = self.att(sa_avgpool2_out)

        # 展平
        conv2_out = conv2_out.view(-1, 32768)
        att_out = att_out.view(-1, 32768)
        ca_out = ca_out.view(-1, 32768)

        att_out = att_out*conv2_out
        out = ca_out + att_out
        # fc+sig
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

        self.fc = nn.Linear(int(feedback_bits // self.B), 32256)

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 16, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock(in_planes=16, mid_planes=128)),
            ("CRBlock2", CRBlock(in_planes=16, mid_planes=128)),
            ("CRBlock3", CRBlock(in_planes=16, mid_planes=128)),
            # ("CRBlock4", CRBlock(in_planes=16, mid_planes=128)),
            # ("CRBlock5", CRBlock(in_planes=16, mid_planes=128)),
            # ("CRBlock6", CRBlock(in_planes=16, mid_planes=128)),
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = conv3x3(16, 2)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits // self.B))
        out = self.fc(out)
        out = out.view(-1, 2, 126, 128)
        out = self.decoder_feature(out)
        out = self.out_cov(out)

        out = self.sig(out)
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
