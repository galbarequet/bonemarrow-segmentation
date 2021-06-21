from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from unet import UNet



def createDeepLabv3(outputchannels=2):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model


class HannahMontanaNet(nn.Module):

    def __init__(self, out_channels=2):
        super(HannahMontanaNet, self).__init__()

        self.out_channels = out_channels #maybe remove
        self.deepLav = createDeepLabv3(out_channels)
        self.unet = UNet(out_channels=out_channels)
        self.conv1 = nn.Conv2d(
            in_channels=out_channels*2, out_channels=out_channels, kernel_size=11, padding=5
        )

    def forward(self, x):
        out_deep = self.deepLav(x)['out']
        unet_out = self.unet(x)
        out = torch.cat((out_deep, torch.tanh(unet_out)), dim=1)
        out = self.conv1(out)

        return torch.sigmoid(out)
