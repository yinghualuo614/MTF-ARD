# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from ...layers import (
    IBN,
    SELayer,
    get_norm,
)
from .build import BACKBONE_REGISTRY
from ...utils import comm
from ...utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}

class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()


        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)


        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x

class Separation(torch.nn.Module):
    def __init__(self, size, num_channel=128, tau=0.1):
        super(Separation, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat): # feat [128,2048,16,8]
        rob_map = self.sep_net(feat) # rob_map [128,2048,16,8]

        mask = rob_map.reshape(rob_map.shape[0], 1, -1) # mask [128,1,262144]
        mask = torch.nn.Sigmoid()(mask) # mask [128,1,262144]
        mask = GumbelSigmoid(tau=self.tau)(mask)   # mask [128,2,262144]
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W)    # mask [128,2048,16,8]

        r_feat = feat * mask    # r_feat [128,2048,16,8]
        nr_feat = feat * (1 - mask) # nr_feat [128,2048,16,8]

        return r_feat, nr_feat, mask


class Recalibration(nn.Module):
    def __init__(self, size, num_channel=64):
        super(Recalibration, self).__init__()
        C, H, W = size
        self.rec_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, nr_feat, mask):
        rec_units = self.rec_net(nr_feat)
        rec_units = rec_units * (1 - mask)
        rec_feat = nr_feat + rec_units

        return rec_feat

#定义18层和34层的残差结构
class BasicBlock(nn.Module):
    # expansion对应我们残差结构中主分支所采用的卷积核个数有没有发生变化
    expansion = 1

    # downsample下采样参数，这个参数对应虚线的残差结构，因为每一层的第一个残差结构有降维的作用
    def __init__(self,
                 inplanes,
                 planes,
                 bn_norm,
                 with_ibn=False,
                 with_se=False,
                 stride=1,
                 downsample=None,
                 reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # IBN是由Instance Normalization（IN）和Batch Normalization构成，有增强学习和泛化能力的作用；
        # IN常用于风格迁移任务中，BN和IN的区别在于BN用的mean和variance是从一个batch中所有的图片统计的，而IN的mean和variance是从单张图片中统计的
        # 使用IBN，还是BN
        # 使用BN的同时，卷积层中的参数bias设置为false，BN层放在卷积层和relu层中间
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction) # 是否使用SE模块，非原创
        else:
            self.se = nn.Identity() # 输入是啥，直接给输出，不做任何的改变。
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.relu(x) # 相比于backbone_resnet,它将这句代码从最后面提到最前面了
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out) # 是否使用SE模块，如果没使用，啥变化也没有

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out

#定义50层和101层的残差结构
class Bottleneck(nn.Module):
    # 在50层，101层和152层的残差结构中，第三层卷积层的卷积核个数是第一，二层卷积层的卷积核个数的四倍，所以expansion设置为4
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.relu(x)   # 相比于backbone_resnet,它将这句代码从最后面提到最前面了
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


# 骨干网（Backbone），包括主干网的选择（如ResNet,ResNest,ResNeXt等）和可以增强主干网表达能力的特殊模块（如non-local、instance batch normalization (IBN)模块等）；
# 骨干网络是推断图像特征图的网络，如没有最后average pooling layer的 ResNet
# 相比于backbone_resnet，它直接不使用non-local模块了，都不去判断是否要使用non-local模块了
class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers, tau=0.1, image_size=(256, 128)):
        self.tau = tau
        self.image_size = image_size

        self.channel_nums = []
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)

        self.separation_resnt18 = Separation(size=(512, int(self.image_size[0] / 16), int(self.image_size[1] / 16)), tau=self.tau)
        # self.recalibration_resnt18 = Recalibration(size=(512, int(self.image_size[0] / 16), int(self.image_size[1] / 16)))
        self.aux_resnt18 = nn.Sequential(nn.Linear(512, 751))

        self.separation_resnt50 = Separation(size=(2048, int(self.image_size[0] / 16), int(self.image_size[1] / 16)),tau=self.tau)
        # self.recalibration_resnt50 = Recalibration(size=(2048, int(self.image_size[0] / 16), int(self.image_size[1] / 16)))
        self.aux_resnt50 = nn.Sequential(nn.Linear(2048, 751))

        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        self.channel_nums.append(self.inplanes)
        return nn.Sequential(*layers)

    def forward(self, x, is_divide_feature = "不从特征中提取鲁棒特征", resnet = "18"):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 不从特征中提取鲁棒特征
        if is_divide_feature == "不从特征中提取鲁棒特征":
            x = F.relu(x, inplace=False)  # 相比于backbone_resnet，它多出了这个relu  [128, 2048, 16, 8]
            return x
        # 将特征分我鲁棒特征和非鲁棒特征
        else:
            if resnet == "18":
                r_feat, nr_feat, mask = self.separation_resnt18(x)
                r_output = self.aux_resnt18(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
                r_feat = F.relu(r_feat, inplace=False)  # 相比于backbone_resnet，它多出了这个relu  [128, 2048, 16, 8]
            elif resnet == "50":
                r_feat, nr_feat, mask = self.separation_resnt50(x)
                r_output = self.aux_resnt50(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
                r_feat = F.relu(r_feat, inplace=False)  # 相比于backbone_resnet，它多出了这个relu  [128, 2048, 16, 8]

            return r_feat, r_output

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            logger.info("ResNet unknown block error!")
        return [bn1, bn2, bn3, bn4]

    def extract_feature(self, x, preReLU=False, is_feature_list = "从多个特征中提取鲁棒特征", resnet = "18"):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if is_feature_list == "从多个特征中提取鲁棒特征":
            if resnet == "18":
                r_feat, nr_feat, mask = self.separation_resnt18(x)
                r_output = self.aux_resnt18(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
                r_feat = F.relu(r_feat, inplace=False)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)


        return [feat1, feat2, feat3, feat4], F.relu(feat4)

    def get_channel_nums(self):
        return self.channel_nums


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_test_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model
