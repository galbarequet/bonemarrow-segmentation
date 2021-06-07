import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataset import BoneMarrowDataset as Dataset
from transform import RandomCrop, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch

from skimage.io import imread
from unet import UNet
from matplotlib import pyplot as plt

from hannahmontananet import HannahMontanaNet

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import time

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

device = 'cpu'

#unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
#unet = createDeepLabv3()
unet = HannahMontanaNet()
state_dict = torch.load('weights/latest_unet.pt', map_location=device)
unet.load_state_dict(state_dict)
unet.eval()
unet.to(device)

crop_size = 576

valid = Dataset(
        images_dir='data_samples',
        subset="validation",
        image_size=0,
        random_sampling=False,
        #transform=transforms(crop=crop_size, color_applay=None)

    )

loader_valid = DataLoader(
        valid,
        batch_size=1,
        drop_last=False,
)
while True:
    for i, batch in enumerate(loader_valid):
        if i == 0:
            continue
        x, y_true = batch
        """
        print(x.shape == y_true.shape)
    
        x = x[0].cpu()
        plt.imshow(x.long().numpy().transpose((1, 2, 0)))
        plt.show()
        y_true = y_true[0].argmax(dim=0).cpu().unsqueeze(0)
        plt.imshow(y_true.long().numpy().transpose((1, 2, 0)))
        plt.show()
        """

        x = x.to(device)
        """
        pred = unet(x)
        pred = pred.cpu()
        #pred = torch.round(pred)
        #pred = pred[0]*100
        pred = pred[0]
        """

        #pred = torch.cat((pred, torch.zeros((1, crop_size,crop_size))))
        x = x[0].cpu()
        plt.imshow(x.long().numpy().transpose((1, 2, 0)))
        plt.show()
        y_true = y_true[0]*100
        y_true = torch.cat((y_true, torch.zeros((1, y_true.shape[1],y_true.shape[2]))))
        plt.imshow(y_true.long().numpy().transpose((1, 2, 0)))
        plt.show()
        print(x.shape[1]%32)
        print(x.shape[2] % 32)
        #plt.imshow(pred.detach().numpy().transpose((1, 2, 0)))
        #plt.show()


