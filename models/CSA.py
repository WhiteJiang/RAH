# -*- coding: utf-8 -*-
# @Time    : 2022/10/30
# @Author  : White Jiang
import math

import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        # 返回通道方向的最大值和平均值
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        # 2维卷积
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        # b 1 h w
        scale = F.sigmoid(x_out)  # broadcasting
        # b c h w
        return scale
        # return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        # 256
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),  # b c
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # b 256 / 16
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)  # b 16->256
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # print(avg_pool.size())
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # b c 1 1
                channel_att_raw = self.mlp(lp_pool)
                # b c
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # b c 1 1 -> b c h w
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor=3, dropout=0.5):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        # expansion 3
        self.mlp = MLP(num_patches, expansion_factor, dropout)
        self.image_size = image_size
        self.spatial_att = SpatialAttention()

    def SpatialGate_forward(self, x):
        # b h*w c
        residual = x
        BB, HH_WW, CC = x.shape
        HH = WW = int(math.sqrt(HH_WW))
        x = x.reshape(BB, CC, HH, WW)
        x = self.spatial_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        # b h*w c
        residual = x
        x_pre_norm = self.SpatialGate_forward((x))
        # b h_w c
        x = self.norm(x_pre_norm)
        # b c h_w
        x = x.transpose(1, 2)
        # b c 14*14
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        # b h*w c
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        """
        Args:
            num_features: 通道数 256
            image_size: 图像尺寸
            num_patches:
            expansion_factor: 3
            dropout: 0.5
        """
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        # 256
        self.mlp = MLP(num_features, expansion_factor, dropout)
        self.image_size = image_size
        self.channel_att = ChannelAttention(num_features, )

    def ChannelGate_forward(self, x):
        residual = x
        BB, HH_WW, CC = x.shape
        HH = WW = int(math.sqrt(HH_WW))
        x = x.reshape(BB, CC, HH, WW)
        # b c h w
        x = self.channel_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.ChannelGate_forward(x)
        x = self.norm(x_pre_norm)
        # b h_w c
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features=196, image_size=14, num_patches=1024, expansion_factor=3, dropout=0.5, CA=False,
                 SA=False):
        super().__init__()
        self.SA = SA
        if SA is True:
            self.token_mixer = TokenMixer(
                num_patches, image_size, num_features, expansion_factor, dropout
            )
        self.CA = CA
        if CA is True:
            self.channel_mixer = ChannelMixer(
                num_patches, image_size, num_features, expansion_factor, dropout
            )

    def forward(self, x):
        BB, CC, HH, WW = x.shape
        # b h w c
        patches = x.permute(0, 2, 3, 1)
        # b h*w c
        patches = patches.view(BB, -1, CC)
        # print(patches.size())
        # 16 49 2048
        # x.shape == (batch_size, num_patches, num_features)
        # b h*w c
        if self.SA is True:
            patches = self.token_mixer(patches)
        if self.CA is True:
            patches = self.channel_mixer(patches)
        embedding_rearrange = patches.reshape(BB, CC, HH, WW)
        embedding_final = embedding_rearrange + x
        # x.shape == (batch_size, num_patches, num_features)
        return embedding_final


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=7)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # residual = x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y1 = self.fc(y1).view(b, c, 1, 1)
        y = y + y1
        scale = y.expand_as(x)
        x = x * scale
        # x = residual + x
        return x


class SALayer(nn.Module):
    def __init__(self):
        super(SALayer, self).__init__()
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.sa(x)
        x = residual + x
        return x


class SA_FBSM(nn.Module):
    def __init__(self):
        super(SA_FBSM, self).__init__()
        self.sa = SpatialAttention()

        self.reset_params()

    def reset_params(self):
        weights_init_kaiming(self.sa)

    def forward(self, x):
        scale = self.sa(x)
        b, c, h, w = scale.size()
        scale = scale.view(b, c, h * w)
        # print(scale)
        # print(scale.size())
        scale_max = torch.max(scale, dim=-1, keepdim=True)[0]
        # scale = scale.view(b, c, h, w)
        scale_suppress = torch.clamp((scale < scale_max * 0.95).float(), min=0.0)
        # print(x)
        scale_suppress = scale_suppress.view(b, c, h, w)
        # print(scale_suppress.size())
        x_suppress = x * scale_suppress
        # print(x_suppress)
        return x, x_suppress


class SELayer_MLP(nn.Module):
    def __init__(self, channel, reduction=1, num_features=2048, expansion_factor=3, dropout=0.5):
        super(SELayer_MLP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=7)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout)

        self.reset_params()

    def reset_params(self):
        weight_init_classifier(self.fc)
        weight_init_classifier(self.mlp)

    def forward(self, x):
        residual = x
        b, c, h, w = x.size()
        # channel attention
        y = self.avg_pool(x).view(b, c)
        y1 = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y1 = self.fc(y1).view(b, c, 1, 1)
        y = y + y1
        scale = y.expand_as(x)
        x = x * scale
        # mlp
        x = residual + x
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, -1, c)
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x.reshape(b, c, h, w)
        out = out + residual
        return out


class CA_SE(nn.Module):
    def __init__(self, in_channels=2048):
        super(CA_SE, self).__init__()
        out_channels = int(in_channels / 2)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_q = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, bias=False)
        self.conv_z = nn.Conv2d(out_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # self.conv_up = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
        #     nn.LayerNorm([out_channels // 4, 1, 1]),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels // 4, in_channels, kernel_size=1)
        # )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        self.reset_params()

    def reset_params(self):
        weights_init_kaiming(self.conv_v)
        weights_init_kaiming(self.conv_q)
        weights_init_kaiming(self.conv_z)
        # weights_init_kaiming(self.conv_up)

    def forward(self, x):
        # residual = x
        feature_v = self.conv_v(x)
        b_v, c_v, h_v, w_v = feature_v.size()
        feature_v = feature_v.view(b_v, c_v, h_v * w_v)

        feature_q = self.conv_q(x)
        b_q, c_q, h_q, w_q = feature_q.size()
        feature_q = feature_q.view(b_q, 1, h_q * w_q)
        feature_q_soft = self.softmax(feature_q)

        scale = torch.matmul(feature_v, feature_q_soft.transpose(1, 2))
        # b C/2 1 1
        scale = scale.unsqueeze(-1)
        scale = self.conv_z(scale)
        # scale = self.conv_up(scale)
        # scale = self.norm(scale)
        scale = self.sigmoid(scale)
        scale = scale.expand_as(x)
        out = x * scale
        # out = residual + out
        return out


if __name__ == '__main__':
    a = torch.rand((2, 4, 7, 7))
    b = torch.rand((2, 4, 7, 7))
    # print(a)
    # _, ids = torch.sort(a, -1, descending=True)
    # print(_)
    # print(ids)
    # print(ids[:, :, :2])
    se = CA_SE
    # print(ac)
    # print(b)
    # print(se(a).size())
