import torch
import torch.nn as nn
from torchvision import models


resnet_model = models.resnet50(pretrained=True)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(ResNetFeatureExtractor, self).__init__()
        size = cfg.INPUT.IMAGE_SIZE
        num_classes = cfg.MODEL.num_classes

        self.features = nn.Sequential(*list(resnet_model.children())[:-2])

    def forward(self):
        x = self.features(x)
        return x