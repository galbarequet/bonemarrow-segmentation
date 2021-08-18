from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.down_skip1 = UNet._skip_block(in_channels, features, name="skip1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.down_skip2 = UNet._skip_block(features, features * 2, name="skip1")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.down_skip3 = UNet._skip_block(features * 2, features * 4, name="skip1")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.down_skip4 = UNet._skip_block(features * 4, features * 8, name="skip1")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.up_skip4 = UNet._skip_block((features * 8) * 2, features * 8, name="skip1")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )

        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.up_skip3 = UNet._skip_block((features * 4) * 2, features * 4, name="skip1")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )

        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.up_skip2 = UNet._skip_block((features * 2) * 2, features * 2, name="skip1")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        self.up_skip1 = UNet._skip_block(features * 2, features, name="skip1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )


    def forward(self, x):
        enc1 = self.encoder1(x)
        skip1 = self.down_skip1(x)
        enc1 = enc1 + skip1

        org_enc2 = self.pool1(enc1)
        enc2 = self.encoder2(org_enc2)
        skip2 = self.down_skip2(org_enc2)
        enc2 = enc2 + skip2

        org_enc3 = self.pool2(enc2)
        enc3 = self.encoder3(org_enc3)
        skip3 = self.down_skip3(org_enc3)
        enc3 = enc3 + skip3

        org_enc4 = self.pool3(enc3)
        enc4 = self.encoder4(org_enc4)
        skip4 = self.down_skip4(org_enc4)
        enc4 = enc4 + skip4

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        org_dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(org_dec4)
        up_skip4 = self.up_skip4(org_dec4)
        dec4 = dec4 + up_skip4

        dec3 = self.upconv3(dec4)
        org_dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(org_dec3)
        up_skip3 = self.up_skip3(org_dec3)
        dec3 = dec3 + up_skip3

        dec2 = self.upconv2(dec3)
        org_dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(org_dec2)
        up_skip2 = self.up_skip2(org_dec2)
        dec2 = dec2 + up_skip2

        dec1 = self.upconv1(dec2)
        org_dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(org_dec1)
        up_skip1 = self.up_skip1(org_dec1)
        dec1 = dec1 + up_skip1

        return self.conv(dec1)

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
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _skip_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True))
                    ]
            ))
