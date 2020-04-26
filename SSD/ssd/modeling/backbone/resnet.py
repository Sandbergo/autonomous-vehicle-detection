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
        # extra block implementation taken from
        # https://github.com/pytorch/vision/blob/7b60f4db9707d7afdbb87fd4e8ef6906ca014720/torchvision/models/resnet.py#L35

        resnet = models.resnet34(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)

        # To run 640x480: Stride = 2 on line 24

        self.resnet = nn.Sequential(*list(resnet.children())[:8], nn.Sequential(nn.Conv2d(512,512,kernel_size=5, stride=3, padding=24, dilation=1)))
        # self.resnet = nn.Sequential(*list(resnet.children())[:8], nn.Sequential(nn.Conv2d(512,512,kernel_size=1, stride=2, padding=0), nn.Upsample(size=(30, 40), mode='bilinear')))
        # self.resnet = nn.Sequential(*list(resnet.children())[:8], nn.Sequential(nn.Conv2d(512,512,kernel_size=5, dilation=5, stride=1, padding=1)))
        # self.resnet = nn.Sequential(*list(resnet.children())[:8], nn.Sequential(nn.Conv2d(512,512,kernel_size=3, stride=2, padding=0)), nn.Sequential(nn.Conv2d(512,512,kernel_size=5, stride=1, dilation = 5, padding=0)))

        print(*list(resnet.children()))


        # Freeze layers
        # ct = 0
        # for child in resnet.children():
        #     ct += 1
        #     if ct < 4:
        #         for param in self.resnet.parameters():
        #             param.requires_grad = False


        self.resnet[3] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0) # hukk old

        # To run 640x480; Change index below from -1 to -2. Because we added the extra conv layer above
        conv4_block1 = self.resnet[-2][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)


        self.additional_layers = self.add_additional_layers()

        """
        resnet.relu = nn.ELU(inplace=True)
        resnet.layer1[0].relu = nn.ELU(inplace=True)
        resnet.layer1[1].relu = nn.ELU(inplace=True)
        resnet.layer1[2].relu = nn.ELU(inplace=True)
        resnet.layer2[0].relu = nn.ELU(inplace=True)
        resnet.layer2[1].relu = nn.ELU(inplace=True)
        resnet.layer2[2].relu = nn.ELU(inplace=True)
        resnet.layer2[3].relu = nn.ELU(inplace=True)
        resnet.layer3[0].relu = nn.ELU(inplace=True)
        resnet.layer3[1].relu = nn.ELU(inplace=True)
        resnet.layer3[2].relu = nn.ELU(inplace=True)
        resnet.layer3[3].relu = nn.ELU(inplace=True)
        resnet.layer3[4].relu = nn.ELU(inplace=True)
        resnet.layer3[5].relu = nn.ELU(inplace=True)
        resnet.layer4[0].relu = nn.ELU(inplace=True)
        resnet.layer4[1].relu = nn.ELU(inplace=True)
        resnet.layer4[2].relu = nn.ELU(inplace=True)
        """

    def add_additional_layers(self):
        layers = nn.ModuleList()


        """
        for i in range(len(self.output_feature_size) - 2):
            layers.append(nn.Sequential(
                 BasicBlock(inplanes = self.output_channels[i], planes = self.output_channels[i+1], stride=2)
            ))
        layers.append(nn.Sequential(
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[-2]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[-2]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[-2]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-1], kernel_size=(2,3), stride=2, padding=0)
        ))
        """
        #layers.append(nn.Sequential(
        #    nn.Conv2d(self.output_channels[0], self.output_channels[0], kernel_size=1, stride=2, padding=0)
        #))
        #print("hey")
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
            nn.BatchNorm2d(self.output_channels[-2]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[-2]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=False),
            nn.BatchNorm2d(self.output_channels[-2]),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-1], kernel_size=(2,3), stride=2, padding=0)
        ))


        return layers

    def forward(self, x):
        out_ch = self.output_channels
        out_feat = self.output_feature_size
        feature_map_size_list = [
            torch.Size([out_ch[0], out_feat[0][1], out_feat[0][0]]),
            torch.Size([out_ch[1], out_feat[1][1], out_feat[1][0]]),
            torch.Size([out_ch[2], out_feat[2][1], out_feat[2][0]]),
            torch.Size([out_ch[3], out_feat[3][1], out_feat[3][0]]),
            torch.Size([out_ch[4], out_feat[4][1], out_feat[4][0]]),
            torch.Size([out_ch[5], out_feat[5][1], out_feat[5][0]])]

        x = self.resnet(x)
        # print("features: ", x.shape)
        features = [x]
        for layer in self.additional_layers:
            x = layer(x)
            features.append(x)

        for idx, feature in enumerate(features):
            out_channel = self.output_channels[idx]
            expected_shape = feature_map_size_list[idx]
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        return tuple(features)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()

        downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            norm_layer(planes),
        )

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

def make_layer(self, block, planes, blocks, stride=1, dilate=False):
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






@registry.BACKBONES.register('resnet')
def resnet(cfg, pretrained=True):
    model = ResNet(cfg)

    return model
