import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from ssd.modeling import registry

class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        # base implementation taken from:
        # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py

        resnet = models.resnet34(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)
        self.resnet = nn.Sequential(*list(resnet.children())[:7])

        conv4_block1 = self.resnet[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

        self.additional_layers = self.add_additional_layers()

    def add_additional_layers(self):
        layers = nn.ModuleList()
        
        for i in range(len(self.output_feature_size) - 2):
            layers.append(nn.Sequential(
                nn.ELU(inplace=False),
                nn.BatchNorm2d(self.output_channels[i]),
                nn.Conv2d(self.output_channels[i], self.output_channels[i], kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=False),
                nn.BatchNorm2d(self.output_channels[i]),
                nn.Conv2d(self.output_channels[i], self.output_channels[i], kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=False),
                nn.BatchNorm2d(self.output_channels[i]),
                nn.Conv2d(self.output_channels[i], self.output_channels[i + 1], kernel_size=3, stride=2, padding=1)
            ))
        layers.append(nn.Sequential(
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[i]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[i]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[i]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-1], kernel_size=(2,3), stride=2, padding=0)
        ))

        return layers

    def forward(self, x):
        x = self.resnet(x)
        features = [x]
        for layer in self.additional_layers:
            x = layer(x)
            features.append(x)

        return tuple(features)


@registry.BACKBONES.register('resnet')
def resnet(cfg, pretrained=True):
    model = ResNet(cfg)
    
    return model