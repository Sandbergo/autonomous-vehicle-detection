import torch
import torch.nn as nn
from torchvision import models


resnet_model = models.resnet50(pretrained=True)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(ResNetFeatureExtractor, self).__init__()
        size = cfg.INPUT.IMAGE_SIZE
        num_classes = cfg.MODEL.NUM_CLASSES 

        for param in resnet_model.parameters():
            param.requires_grad = False

        num_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_features, num_classes)

        # self.features = nn.Sequential(*list(resnet_model.children())[:-2])
        self.features = nn.ModuleList(list(resnet_model.children())[:-2])

    def forward(self, x):
        # x = self.features(x)
        # return x

        features = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i % 8 == 0:
                features.append(x)

        return tuple(features)
