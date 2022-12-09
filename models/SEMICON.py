from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from models.resnet_torch import *
import math
from models.CSA import ChannelAttention, SpatialAttention, SA_FBSM, CA_SE
from models.CRFC import CRA_DOT

# from ..utils.serialization import load_checkpoint, copy_state_dict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'SEMICON_backbone']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PosAttention(nn.Module):
    def __init__(self, in_channels=2048, SA=None, CA=None):
        super(PosAttention, self).__init__()
        hidden_channels = int(in_channels / 4)
        self.SA = None
        self.CA = None
        if SA is not None:
            self.SA = SpatialAttention()
            self.qnorm1 = nn.BatchNorm2d(hidden_channels)
        if CA is not None:
            self.CA = ChannelAttention()
            self.qnorm2 = nn.BatchNorm2d(hidden_channels)
        self.qconv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.qconv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.qnorm3 = nn.BatchNorm2d(in_channels)
        self.qrelu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.qconv1(x)
        if self.SA is not None:
            t = self.SA(y)
            t = t + y
            # print(t.size())
            y = self.qnorm1(t)
        if self.CA is not None:
            t = self.CA(y)
            t = t + y
            y = self.qnorm2(t)
        out = self.qconv2(y)
        out = self.qnorm3(out)
        out = self.qrelu(out)
        return out


"""
Two stage
"""


class ChannelTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        # 将qkv做映射的卷积
        # 分组卷积得到N个块 每个块d维
        # 分组卷积的作用类似于在限定区间内做映射
        self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=num_heads)
        # d个块,每个块N维
        self.qkv2 = nn.Conv2d(dim, dim * 3, 1, groups=head_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 将输入映射为 q k v  在通道方向均等划分为num_head个特征向量
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 第一阶段的注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 将注意力与原值相加
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        # 做softmax
        attn = attn.softmax(dim=-1)
        # 残差链接 同时转换成d个n块
        x = ((attn @ v).reshape(B, C, H, W) + x).reshape(B, self.num_heads, self.head_dim, H, W).transpose(1,
                                                                                                           2).reshape(B,
                                                                                                                      C,
                                                                                                                      H,
                                                                                                                      W)
        y = self.norm(x)
        x = self.relu(y)
        # 第二阶段的注意力计算
        qkv2 = self.qkv2(x).reshape(B, 3, self.head_dim, self.num_heads, H * W).transpose(0, 1)
        q, k, v = qkv2[0], qkv2[1], qkv2[2]

        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, self.head_dim, self.num_heads, H, W).transpose(1, 2).reshape(B, C, H, W) + y
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 输出的中间通道数
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# 设计的backbone只有三层 第四层用于获取局部和全局特征
def SEMICON_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls['resnet50'],
        #                                       progress=progress)
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class Refine_Post(nn.Module):
    def __init__(self):
        super(Refine_Post, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        return pool_x


class ResNet_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1

        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        # self.attention = PosAttention(in_channels=1024, SA=True, CA=True)
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # layers.append(SALayer())
        if self.is_local:
            # layers.append(SELayer(2048))
            layers.append(CA_SE())
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            # if _ == 1:
            #     layers.append(SELayer(2048))
            # layers.append(PosAttention(SA=True, CA=True))
            # 做通道交互
            # layers.append(ChannelTransformer(planes * block.expansion, max(planes * block.expansion // 64, 16)))
        # layers.append(nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1))
        # layers.append(MixerLayer(SA=True, CA=True))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x = self.attention(x)
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls['resnet50'],
        #                                       progress=progress)
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict, strict=False)
    return model


class SEM(nn.Module):

    def __init__(self, block, layer, att_size=4, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SEM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.att_size = att_size
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        # 相当于产生的每个通道相对于原始输入为64通道
        self.base_width = width_per_group
        # 首先生成第四层特征，且步长1，分辨率相较于全局特征要大
        self.layer4 = self._make_layer(block, 512, layer, stride=1)
        # 生成一通道的张量
        self.feature1 = nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.feature2 = nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.feature3 = nn.Sequential(
            conv1x1(self.inplanes, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        att_expansion = 0.25
        layers = []
        # 输出通道减少至1/4 输入1024 输出1024 base_width=64 groups=1
        layers.append(block(self.inplanes, int(self.inplanes * att_expansion), stride,
                            downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        # l = 1: 首先1024 -> 256, self.inplanes = 256,  则最后l=1的最后输出为256
        # 同理l = 2时 最后输出为64通道
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, int(self.inplanes * att_expansion)),
                nn.BatchNorm2d(int(self.inplanes * att_expansion))
            ))
            self.inplanes = int(self.inplanes * att_expansion)
            layers.append(block(self.inplanes, int(self.inplanes * att_expansion), groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _mask(self, feature, x):
        with torch.no_grad():
            # 计算通道方向的均值
            cam1 = feature.mean(1)
            # 尺寸修改为 B H W大小
            attn = torch.softmax(cam1.view(x.shape[0], x.shape[2] * x.shape[3]), dim=1)  # B,H,W
            # 计算均值方差
            std, mean = torch.std_mean(attn)
            # 利用均值方差更新特征图上的值
            attn = (attn - mean) / (std ** 0.3) + 1  # 0.15
            # 继续将值还原为 BS 1 H W 并将值约束到 0 2区间
            attn = (attn.view((x.shape[0], 1, x.shape[2], x.shape[3]))).clamp(0, 2)
        return attn

    def _forward_impl(self, x):
        # bs*64*14*14
        x = self.layer4(x)
        # bs*1*14*14， 第一层的M矩阵
        fea1 = self.feature1(x)
        # 第一层的SEM矩阵
        attn = 2 - self._mask(fea1, x)
        # 将通道从1扩展到1024将其与未施加注意力图做点乘，得到用于第二层的抑制后的输入P2
        x = x.mul(attn.repeat(1, self.inplanes, 1, 1))
        # 计算第二层的T heat矩阵
        fea2 = self.feature2(x)
        # 第二层的SEM矩阵
        attn = 2 - self._mask(fea2, x)
        # 第三层的输入 P3
        x = x.mul(attn.repeat(1, self.inplanes, 1, 1))
        # 生成的第三层M矩阵
        fea3 = self.feature3(x)
        # 串联成三通道的特征 三通道的注意力图
        x = torch.cat([fea1, fea2, fea3], dim=1)
        # 返回三个层的M矩阵
        return x

    def forward(self, x):
        return self._forward_impl(x)


# 生成3通道的注意力图，分辨率未变
def SEMICON_attention(att_size=3, pretrained=False, progress=True, **kwargs):
    # 实际上不加载预训练参数, att_size并没有提供任何作用
    model = SEM(Bottleneck, 3, att_size=att_size, **kwargs)

    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls['resnet50'],
        #                                       progress=progress)
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        for name in list(state_dict.keys()):
            if 'fc' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


"""
Visual
"""


class SEMICON(nn.Module):
    def __init__(self, code_length=12, num_classes=200, att_size=3, feat_size=2048, device='cpu', pretrained=True):
        super(SEMICON, self).__init__()
        # 骨干网
        self.backbone = SEMICON_backbone(pretrained=pretrained)
        # 用于生成全局特征。 最后只返回一个bs c h*w的特征 全局级别的转换网络
        self.refine_global = SEMICON_refine(is_local=False, pretrained=pretrained)
        # 用于生成局部特征 局部特征会返回未经池化的特征bs c h w和bs c h*w
        # SEMICON_refine包含ICON的通道交互 局部级别的转换网络
        self.refine_local = SEMICON_refine(pretrained=pretrained)
        # 用于生成局部特征的注意力, 最后生成3通道的注意力图 多层注意力累积 生成注意力的网络
        # self.attention = SEMICON_attention(att_size=att_size)
        self.attention = SA_FBSM()
        self.cf = CRA_DOT()
        self.cls = nn.Linear(2048, 101)
        self.cls_loc = nn.Linear(2048, 101)
        # self.attention = SALayer()
        # 哈希激活代码
        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length
        # 12 24 32 48
        # global
        if self.code_length != 32:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2, feat_size))
            torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        else:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2 + 1, feat_size))
            torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        # self.W_G = nn.Parameter(torch.Tensor(code_length // 2, feat_size))
        # torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        # self.W_G = nn.Parameter(torch.Tensor(code_length // 2, feat_size))
        # torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        # local
        self.W_L1 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))
        self.W_L2 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L2, a=math.sqrt(5))
        self.W_L3 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L3, a=math.sqrt(5))
        # 伯努利分布
        self.bernoulli = torch.distributions.Bernoulli(0.5)
        self.device = device

    def forward(self, x):
        # 提取特征
        out = self.backbone(x)  # .detach()
        # 生成全局特征,基于第三层特征生成
        global_f = self.refine_global(out)
        # for local
        out_local1, input_local2 = self.attention(out)
        out_local2, input_local3 = self.attention(input_local2)
        out_local3, _ = self.attention(input_local3)
        # _, out_local1 = self.attention(out)
        # _, out_local2 = self.attention(out_local1)
        # _, out_local3 = self.attention(out_local2)
        # print(out_local2.size())
        # 后续的ICON处理模块
        # bs 1024 14 14 -> bs 2048 1 1
        # CRA_DOT
        r2_r1, r1_r2 = self.cf(out_local2, out_local1)
        r3_r2, r2_r3 = self.cf(out_local3, out_local2)
        # r3_r1, r1_r3 = self.cf(out_local3, out_local1)
        out_local1 = out_local1 + r1_r2
        out_local2 = out_local2 + 0.5 * (r2_r1 + r2_r3)
        out_local3 = out_local3 + r3_r2
        # r2_r1, r1_r2 = self.cf(out, out)
        # r3_r2, r2_r3 = self.cf(out, out)
        # r3_r1, r1_r3 = self.cf(out, out)

        local_f1, avg_local_f1 = self.refine_local(out_local1)
        local_f2, avg_local_f2 = self.refine_local(out_local2)
        local_f3, avg_local_f3 = self.refine_local(out_local3)

        # 特征映射到哈希码上
        deep_S_G = F.linear(global_f, self.W_G)

        deep_S_1 = F.linear(avg_local_f1, self.W_L1)
        deep_S_2 = F.linear(avg_local_f2, self.W_L2)
        deep_S_3 = F.linear(avg_local_f3, self.W_L3)
        # 一个全局和三个局部特征映射 生成最后的哈希编码 组成发最后的哈希码
        deep_S = torch.cat([deep_S_G, deep_S_1, deep_S_2, deep_S_3], dim=1)
        # 哈希激活层
        ret = self.hash_layer_active(deep_S)
        if self.training:
            cls = self.cls(global_f)
            cls1 = self.cls_loc(avg_local_f1)
            cls2 = self.cls_loc(avg_local_f2)
            cls3 = self.cls_loc(avg_local_f3)
        #     return ret, local_f1, cls, cls1, cls2
            return ret, local_f1, cls , cls1, cls2, cls3

        return ret, local_f1


def semicon(code_length, num_classes, att_size, feat_size, device, pretrained=False, **kwargs):
    # 实际att_size = 1
    model = SEMICON(code_length, num_classes, att_size, feat_size, device, pretrained, **kwargs)
    return model


if __name__ == '__main__':
    var1 = torch.FloatTensor(1, 1024, 14, 14)
    var2 = torch.FloatTensor(1, 1024, 14, 14)
    result = (var1 @ var2.transpose(-2, -1))
    print(result.shape)
