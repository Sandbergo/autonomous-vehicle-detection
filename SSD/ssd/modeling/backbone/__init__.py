from ssd.modeling import registry
from .vgg import VGG
from .resnet import ResNet

__all__ = ['build_backbone', 'VGG', 'ResNet']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
