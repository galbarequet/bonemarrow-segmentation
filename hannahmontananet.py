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
    model = models.segmentation.deeplabv3_resnet101(pretrained=False,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model


class HannahMontanaNet(nn.Module):

    def __init__(self, out_channels=4, deeplav_features=16, unet_features=16, post_processing_features=64):
        super(HannahMontanaNet, self).__init__()

        self.out_channels = out_channels #maybe remove
        self.deepLav = createDeepLabv3(deeplav_features)
        self.unet = UNet(out_channels=unet_features)

        self.conv1 = HannahMontanaNet._block(
            in_channels=deeplav_features + unet_features, features=post_processing_features, name='conv1'
        )

        self.conv2 = HannahMontanaNet._block(
            in_channels=post_processing_features, features=post_processing_features, name='conv1'
        )

        self.conv3 = HannahMontanaNet._block(
            in_channels=post_processing_features, features=post_processing_features, name='conv1'
        )

        self.final_conv = nn.Conv2d(in_channels=post_processing_features, out_channels=out_channels,
                                    kernel_size=3, padding=1)

    def forward(self, x):
        out_deep = self.deepLav(x)['out']
        unet_out = self.unet(x)

        out = torch.cat((out_deep, torch.tanh(unet_out)), dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.final_conv(out)

        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )