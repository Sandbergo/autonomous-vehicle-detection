import torch
import torch.nn as nn
from torchvision import models
from ssd.config.defaults import cfg


inception_model = models.inception_v3(pretrained=True)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(InceptionFeatureExtractor, self).__init__()
        
        self.size = cfg.INPUT.IMAGE_SIZE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_feature_sizes = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.Conv2d_2a_3x3 = inception_model.Conv2d_2a_3x3  # N x 32 x 149 x 149
        self.Conv2d_3b_1x1 = inception_model.Conv2d_3b_1x1  # N x 64 x 73 x 73
        self.Conv2d_4a_3x3 = inception_model.Conv2d_4a_3x3  # N x 80 x 73 x 73
        self.Mixed_5b = inception_model.Mixed_5b            # N x 192 x 35 x 35
        self.Mixed_6c = inception_model.Mixed_6c            # N x 768 x 17 x 17

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        print('Start of forward.')

        x1 = self.Conv2d_2a_3x3(x)  # N x 32 x 149 x 149
        x2 = self.Conv2d_3b_1x1(x1) # N x 64 x 73 x 73
        x3 = self.Conv2d_4a_3x3(x2) # N x 80 x 73 x 73
        x4 = self.Mixed_5b(x3)      # N x 192 x 35 x 35
        x5 = self.Mixed_6c(x4)      # N x 768 x 17 x 17

        print('End of forward.')

        return [x1, x2, x3, x4, x5], x5

    def forward(self, x):

        assert self.size == 299, \
            f'Expected input image size 299, got {self.size}.'

        inception_features, x5 = self._forward(x)

        return tuple(inception_features)


if __name__ == "__main__":
    my_inception_model = InceptionFeatureExtractor(cfg)

        
