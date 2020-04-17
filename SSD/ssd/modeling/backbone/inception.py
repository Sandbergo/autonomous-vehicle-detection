import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

INCEPTION_IMAGE_SIZE = 299
INCEPTION_END_SIZE = 288

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, cfg, is_pretrained=True, transform_input=False):
        super(InceptionFeatureExtractor, self).__init__()

        self.size = cfg.INPUT.IMAGE_SIZE
        # self.num_classes = cfg.MODEL.NUM_CLASSES
        self.out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.out_feature_sizes = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.transform_input = transform_input

        self.inception_model = models.inception_v3(pretrained=True, transform_input=transform_input)

        # Freeze all layers in the inception_v3 model
        for param in self.inception_model.parameters():
            param.requires_grad = False

        # self.inception_features = nn.Sequential(*list(inception.children())[:-1])

        """

        # input                                             # N x 3 x 299 x 299

        self.Conv2d_1a_3x3 = inception_model.Conv2d_1a_3x3  # N x 32 x 149 x 149
        self.Conv2d_2a_3x3 = inception_model.Conv2d_2a_3x3  # N x 32 x 147 x 147
        self.Conv2d_2b_3x3 = inception_model.Conv2d_2b_3x3  # N x 64 x 147 x 147

        # max_pool2d                                        # N x 64 x 73 x 73

        self.Conv2d_3b_1x1 = inception_model.Conv2d_3b_1x1  # N x 80 x 73 x 73
        self.Conv2d_4a_3x3 = inception_model.Conv2d_4a_3x3  # N x 192 x 71 x 71

        # max_pool2d                                        # N x 192 x 35 x 35

        self.Mixed_5b = inception_model.Mixed_5b            # N x 256 x 35 x 35    x1
        self.Mixed_5c = inception_model.Mixed_5c            # N x 288 x 35 x 35
        self.Mixed_5d = inception_model.Mixed_5d            # N x 288 x 35 x 35
        self.Mixed_6a = inception_model.Mixed_6a            # N x 768 x 17 x 17    x2
        self.Mixed_6b = inception_model.Mixed_6b            # N x 768 x 17 x 17
        self.Mixed_6c = inception_model.Mixed_6c            # N x 768 x 17 x 17    x3
        self.Mixed_6d = inception_model.Mixed_6d            # N x 768 x 17 x 17
        self.Mixed_6e = inception_model.Mixed_6e            # N x 768 x 17 x 17
        self.Mixed_7a = inception_model.Mixed_7a            # N x 1280 x 8 x 8     x4
        self.Mixed_7b = inception_model.Mixed_7b            # N x 2048 x 8 x 8     x5
        self.Mixed_7c = inception_model.Mixed_7c            # N x 2048 x 8 x 8
        
        # self.end_pooling = (
        #   torch.nn.AdaptiveAvgPool2d((1,1))               # N x 2048 x 1 x 1     x6
        # )

        """

        self.extra_layer1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=INCEPTION_END_SIZE, )
        )

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):

        """
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x1 = self.Mixed_5b(x)
        x = x1

        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x2 = self.Mixed_6a(x)
        x = x2

        x = self.Mixed_6b(x)

        x3 = self.Mixed_6c(x)
        x = x3

        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        x4 = self.Mixed_7a(x)
        x = x4

        x5 = self.Mixed_7b(x)
        x = x5

        x = self.Mixed_7c(x)
        """

        x = self.inception_model.Conv2d_1a_3x3(x)
        x = self.inception_model.Conv2d_2a_3x3(x)
        x = self.inception_model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.inception_model.Conv2d_3b_1x1(x)
        x = self.inception_model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x1 = self.inception_model.Mixed_5b(x)
        x = x1

        x = self.inception_model.Mixed_5c(x)
        x = self.inception_model.Mixed_5d(x)

        """

        x2 = self.inception_model.Mixed_6a(x)
        x = x2

        x = self.inception_model.Mixed_6b(x)

        x3 = self.inception_model.Mixed_6c(x)
        x = x3

        x = self.inception_model.Mixed_6d(x)
        x = self.inception_model.Mixed_6e(x)

        x4 = self.inception_model.Mixed_7a(x)
        x = x4

        x5 = self.inception_model.Mixed_7b(x)
        x = x5

        x = self.inception_model.Mixed_7c(x)

        """

        # return [x1, x2, x3, x4, x5], x
        return x                                # N x 288 x 35 x 35

    def _extra_forward(self, x):


    def forward(self, x):

        feature_map_sizes = [
            torch.Size([self.out_channels[0], self.out_feature_sizes[0], self.out_feature_sizes[0]]),
            torch.Size([self.out_channels[1], self.out_feature_sizes[1], self.out_feature_sizes[1]]),
            torch.Size([self.out_channels[2], self.out_feature_sizes[2], self.out_feature_sizes[2]]),
            torch.Size([self.out_channels[3], self.out_feature_sizes[3], self.out_feature_sizes[3]]),
            torch.Size([self.out_channels[4], self.out_feature_sizes[4], self.out_feature_sizes[4]]),
            torch.Size([self.out_channels[5], self.out_feature_sizes[5], self.out_feature_sizes[5]])
        ]

        assert self.size == INCEPTION_IMAGE_SIZE, \
            f'Expected input image size 299, got {self.size}.'

        x = self._transform_input(x)

        inception_features, x_ = self._forward(x)

        x6 = F.adaptive_avg_pool2d(x_, (1,1))

        out_features = inception_features + [x6]

        for i, feature in enumerate(out_features):
            out_channel = self.out_channels[i]
            expected_shape = feature_map_sizes[i]
            assert feature.shape[1:] == expected_shape, \
                f'Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {i}'

        return tuple(out_features)

    def forward2(self, x):

        feature_map_size_list = [
            torch.Size([self.out_channels[0], self.out_feature_sizes[0], self.out_feature_sizes[0]]),
            torch.Size([self.out_channels[1], self.out_feature_sizes[1], self.out_feature_sizes[1]]),
            torch.Size([self.out_channels[2], self.out_feature_sizes[2], self.out_feature_sizes[2]]),
            torch.Size([self.out_channels[3], self.out_feature_sizes[3], self.out_feature_sizes[3]]),
            torch.Size([self.out_channels[4], self.out_feature_sizes[4], self.out_feature_sizes[4]]),
            torch.Size([self.out_channels[5], self.out_feature_sizes[5], self.out_feature_sizes[5]])
        ]

        assert self.size == 299, \
            f'Expected input image size 299, got {self.size}.'

        out_features = []

        x = self.inception_features(x)
        print('SHAPE OF X after inception:', x.shape)  # N x 768 x 17 x 17

        x = F.adaptive_max_pool2d(x, (10, 10))
        print('SHAPE OF X after adaptive pool:', x.shape)  # N x 768 x 17 x 17

        for i, feature in enumerate(out_features):
            out_channel = self.out_channels[i]
            expected_shape = feature_map_size_list[i]
            assert feature.shape[1:] == expected_shape, \
                f'Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {i}'

        return tuple(out_features)


if __name__ == "__main__":
    print('Hello')
    my_inception_model = InceptionFeatureExtractor(None)

        
