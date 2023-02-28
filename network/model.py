from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from functools import reduce
from hr_config import HRNET_18, HRNET_32, HRNET_48

arch_map = {
    "simple": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "res18": {"block": "BASIC", "layers": [2, 2, 2, 2]},
    "res34": {"block": "BASIC", "layers": [3, 4, 6, 3]},
    "res101": {"block": "BOTTLENECK", "layers": [3, 4, 23, 3]},
    "hr18": HRNET_18,
    "hr32": HRNET_32,
    "hr48": HRNET_48,
    "unet": None,
}

BN_MOMENTUM = 0.1
ALIGN_CORNERS = None


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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


class VGGEncoder(nn.Module):
    def __init__(self, image_size, image_channel, features, arch, bn):
        super(VGGEncoder, self).__init__()

        self.bn = bn
        self.arch = arch
        self.image_size = image_size
        self.image_channel = image_channel
        self.features = features

        self.vgg, self.result_shape = self.make_layers()

        result_size = reduce(lambda a, b: a*b, self.result_shape)
        self.encoder = nn.Sequential(
            nn.Linear(result_size, min(result_size, 4096)),
            nn.ReLU(inplace=True),
            nn.Linear(min(result_size, 4096), min(result_size, 4096)),
            nn.ReLU(inplace=True),
            nn.Linear(min(result_size, 4096), self.features)
        )
        self.weight_init()

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

    def make_layers(self):
        layers = []
        result_size = self.image_size
        in_channel = self.image_channel
        for layer_label in self.arch:
            if layer_label == "M":
                # layers.append(nn.Conv2d(in_channel, in_channel,kernel_size=3, stride=2, padding=1))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                result_size //= 2
            else:
                out_channel = int(layer_label)
                conv_2d = nn.Conv2d(in_channel, out_channel,
                                    kernel_size=3, padding=1)
                if self.bn:
                    layers += [conv_2d,
                               nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)]
                else:
                    layers += [conv_2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        return nn.Sequential(*layers), [in_channel, result_size, result_size]

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, out_size=None):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels

        self.pre = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            BasicBlock(out_channels, out_channels)
        )

        if (out_size is not None):
            self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.pre(x)
        if self.out_size is None:
            x = F.interpolate(x, scale_factor=2)
        else:
            x = F.interpolate(x, size=self.out_size)
        return self.conv(x)


class Upsample2(nn.Module):
    def __init__(self, in_channels, out_channels, out_size=None):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels

        self.pre = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            conv3x3(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

        if (out_size is not None):
            self.conv = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x = self.pre(x)
        if self.out_size is None:
            x = F.interpolate(x, scale_factor=2)
        else:
            x = F.interpolate(x, size=self.out_size)
        return self.conv(x)


class SimpleDecoder(nn.Module):
    def __init__(self, image_size, image_channel, features, input_shape, bn):
        super(SimpleDecoder, self).__init__()
        self.bn = bn
        self.image_size = image_size
        self.image_channel = image_channel
        self.features = features
        self.input_shape = input_shape

        input_size = reduce(lambda a, b: a*b, self.input_shape)
        self.reshaper = nn.Sequential(
            nn.Linear(self.features, input_size),
            nn.ReLU(inplace=True),
        )

        res = self.input_shape[1]
        layers = []
        in_channel = self.input_shape[0]
        while res != self.image_size:
            out_channel = self.image_channel if res * \
                2 >= self.image_size else max(in_channel//2, 1)
            if res * 2 >= self.image_size:
                layers.append(
                    Upsample2(in_channel, out_channel, self.image_size))
                res = self.image_size
            else:
                layers.append(Upsample2(in_channel, out_channel))
                res *= 2
            in_channel = out_channel

        self.decoder = nn.Sequential(*layers)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.reshaper(x)
        x = x.view(x.size(0), *self.input_shape)
        x = self.decoder(x)
        return x


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self,
                 cfg,
                 norm_layer=None):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        # stem network
        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        # self.stage4_cfg = cfg['STAGE4']
        # num_channels = self.stage4_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage4_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition3 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        # self.stage4, pre_stage_channels = self._make_stage(
        #     self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=19,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride,
                      downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage3(x_list)
        # y_list = self.stage3(x_list)

        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         if i < self.stage3_cfg['NUM_BRANCHES']:
        #             x_list.append(self.transition3[i](y_list[i]))
        #         else:
        #             x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        # x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = torch.cat([x[0], x1, x2], 1)
        # x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x


class HREncoder(nn.Module):
    def __init__(self, image_size, image_channel, features, arch):
        super().__init__()
        self.image_size = image_size
        self.hr = HighResolutionNet(arch, None)

        self.feature_layers = nn.Sequential(
            nn.Linear(image_size * image_size * 19, features),
            nn.ReLU(inplace=True),
            nn.Linear(features, features)
        )
        self.result_shape = [features, 1, 1]

    def forward(self, x):
        x = self.hr(x)
        x = torch.flatten(x, start_dim=1)
        x = self.feature_layers(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        image_size, image_channel, features, config,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        block = blocks_dict[config['block']]
        layers = config['layers']

        self.image_size = image_size
        result_size = self.image_size
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(image_channel, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        result_size = (result_size+1)//2
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if (self.image_size > 64):
            result_size = (result_size+1)//2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        result_size = (result_size+1)//2
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        result_size = (result_size+1)//2
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        result_size = (result_size+1)//2
        if self.image_size >= 64:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            result_size = 1
        self.result_shape = [512, result_size, result_size]
        self.fc = nn.Linear(
            reduce(lambda a, b: a*b, self.result_shape), features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.image_size > 64:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.image_size >= 64:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        image_size, image_channel, input_shape, features, config,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        block = blocks_dict[config['block']]
        layers = config['layers']

        self.image_size = image_size
        self.groups = groups
        self.base_width = width_per_group

        self.features = features
        self.input_shape = input_shape

        input_size = reduce(lambda a, b: a*b, self.input_shape)
        self.reshaper = nn.Sequential(
            nn.Linear(self.features, input_size)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, image_channel, 1)
        )

        self.layer1 = self._make_layer(block, 64, 32, layers[0])
        self.layer2 = self._make_layer(
            block, 128, 64, layers[1], dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, 128, layers[2], dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, 256, layers[3], dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        in_planes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.reshaper(x).view(x.size(0), *self.input_shape)
        x = F.interpolate(x, size=self.image_size//8)

        x = self.layer4(x)
        x = F.interpolate(x, size=self.image_size//4)

        x = self.layer3(x)
        x = F.interpolate(x, size=self.image_size//2)

        x = self.layer2(x)
        x = F.interpolate(x, size=self.image_size)

        x = self.layer1(x)
        x = self.conv(x)
        return torch.sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class HRDecoder(nn.Module):
    def __init__(self, image_size, image_channel, features, bn, arch):
        super(HRDecoder, self).__init__()
        self.bn = bn
        self.image_size = image_size
        self.image_channel = image_channel
        self.features = features
        self.input_shape = [image_channel, image_size, image_size]

        input_size = reduce(lambda a, b: a*b, self.input_shape)
        # restore original shape from feature space
        self.reshaper = nn.Sequential(
            nn.Linear(self.features, input_size),
            nn.ReLU(inplace=True),
        )

        self.hr = HighResolutionNet(arch_map[arch], None)

        # reduce back to image_channel
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=19,
                out_channels=image_channel,
                kernel_size=1,
                stride=1,
                padding=0),
        )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.reshaper(x)
        x = x.view(x.size(0), *self.input_shape)
        x = self.hr(x)
        x = self.decoder(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, image_size=64):
        super(UNet, self).__init__()

        cur_size = image_size

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
        else:
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        cur_size = cur_size//2

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features*2, kernel_size=2, stride=2
            )
        else:
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features*2, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, name="dec2")
        cur_size = cur_size//2

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
        else:
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, name="dec3")
        cur_size = cur_size//2

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        if (cur_size % 2 == 0):
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2, output_padding=1
            )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, name="dec4")

        self.bottleneck = UNet._block(
            features * 8, features * 16, name="bottleneck")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNetEncoder(nn.Module):
    def __init__(self, image_size, image_channel, features):
        super().__init__()
        self.encoder = UNet(image_channel, image_channel, 32, image_size)
        self.feature_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(image_size * image_size * image_channel, features),
            nn.ReLU(inplace=True),
            nn.Linear(features, features)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.feature_layers(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, image_size, image_channel, features):
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.decoder = UNet(image_channel, image_channel, 32, image_size)
        self.feature_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(features, image_size * image_size * image_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), self.image_channel,
                   self.image_size, self.image_size)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, image_size, image_channel, features, arch, bn):
        super().__init__()
        self.arch = arch
        if arch.startswith('vgg'):
            self.encoder = VGGEncoder(
                image_size, image_channel, features, arch_map[arch], bn)
            encoder_output_shape = self.encoder.result_shape
            self.decoder = SimpleDecoder(
                image_size, image_channel, features, encoder_output_shape, bn)
        elif arch.startswith('res'):
            self.encoder = ResNetEncoder(
                image_size, image_channel, features, arch_map[arch])
            encoder_output_shape = self.encoder.result_shape
            self.decoder = SimpleDecoder(
                image_size, image_channel, features, encoder_output_shape, bn)
        elif arch.startswith('hr'):
            self.encoder = HREncoder(
                image_size, image_channel, features, arch_map[arch])
            encoder_output_shape = self.encoder.result_shape
            self.decoder = HRDecoder(
                image_size, image_channel, features, bn, arch)
        elif arch.startswith('unet'):
            self.encoder = UNet(1, 1, 32)
            self.decoder = lambda x: x
        elif arch.startswith('wnet'):
            self.encoder = UNetEncoder(image_size, image_channel, features)
            self.decoder = UNetDecoder(image_size, image_channel, features)
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)
