from ssd.modeling import registry
from .vgg import VGG
from .basic import BasicModel
from .efficient_net import EfficientNet

__all__ = ['build_backbone', 'VGG', 'BasicModel', 'EfficientNet']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)