import torch
from torch import nn
import logging

from torch.utils import model_zoo
from torchvision.models.resnet import ResNet

__all__ = ['ResNextBlock', 'resnext50', 'resnext101', 'resnext152', 'save']

model_urls = {
    'resnext101' : 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl',
    'resnext152' : 'https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl'
}


class ResNextBlock(nn.Module):
    """
    Class for basic ResNext block. Has architecture of bottleneck of ResNet.
    """
    expansion = 4
    multiplier = 2

    def __init__(self, input_size, output_size, stride=1, downsample=None, cardinality=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=output_size * self.multiplier, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_size * self.multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_size * self.multiplier, out_channels=output_size * self.multiplier,
                      kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality),
            nn.BatchNorm2d(output_size * self.multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_size * self.multiplier, out_channels=output_size * self.expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(output_size * self.expansion),
        )
        self.downsample = downsample

    def forward(self, input):
        logging.debug("ResNext Block forward")
        x = self.conv.forward(input)

        residual = input
        if self.downsample is not None:
            residual = self.downsample.forward(input)

        assert x.size() == residual.size()

        return nn.ReLU(inplace=True)(x + residual)


def resnext50(pretrained=False, **kwargs):
    """
    Constructs a ResNeXt-50 model
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :return: ResNeXt-50 model
    """
    logging.info("Creating ResNeXT-50 model")

    model = ResNet(ResNextBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        logging.error("Pretrained ResNeXt-50 is not implemented yet.")
        raise NotImplementedError("Pretrained ResNeXt-50 not implemented")
        # model.load_state_dict(model_zoo.load_url(model_urls['resnext50']))
    return model


def resnext101(pretrained=False, **kwargs):
    """
    Constructs a ResNeXt-101 model
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :return: ResNeXt -101 model
    """
    logging.info("Creating ResNeXT-101 model")

    model = ResNet(ResNextBlock, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101']))
    return model


def resnext152(pretrained=False, **kwargs):
    """
    Constructs a ResNeXt-152 model
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :return: ResNeXt -152 model
    """
    logging.info("Creating ResNeXT-152 model")

    model = ResNet(ResNextBlock, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext152']))
    return model

def save(path, model):
    """
    Saves trained weights
    :param path: Path to save weights to.
    :param model: Model desired to be saved
    """
    torch.save(model.state_dict(), path)