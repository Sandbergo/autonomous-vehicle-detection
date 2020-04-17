import torch
import torch.nn as nn
from torchvision import models


resnet_model = models.resnet34(pretrained=True)


###---                                      START OF BORROWED CODE                                              ---###

# The implementation of the following class is borrowed from the official torchvision resnet model,
# with a few tweaks.

# Link: 'https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py'

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=1
    ):
        super(Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_relu(out)

        return out


###---                                            END OF BORROWED CODE                                          ---###


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        
        size = cfg.INPUT.IMAGE_SIZE
        num_classes = cfg.MODEL.NUM_CLASSES

        for param in resnet_model.parameters():
            param.requires_grad = False
