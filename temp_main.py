import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataset import BoneMarrowDataset as Dataset
from transform import RandomCrop
from torch.utils.data import DataLoader
import numpy as np
import torch

from skimage.io import imread
from unet import UNet
from matplotlib import pyplot as plt

device = 'cpu'

unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
state_dict = torch.load('weights/unet.pt', map_location=device)
unet.load_state_dict(state_dict)
unet.eval()
unet.to(device)

crop_size = 1440

valid = Dataset(
        images_dir='data_samples',
        subset="validation",
        image_size=0,
        random_sampling=False,
        transform=RandomCrop(crop_size)
    )

loader_valid = DataLoader(
        valid,
        batch_size=1,
        drop_last=False,
)

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
    pred = unet(x)
    pred = pred.cpu()
    pred = torch.round(pred)
    pred = pred[0]*100

    pred = torch.cat((pred, torch.zeros((1, crop_size,crop_size))))
    x = x[0].cpu()
    plt.imshow(x.long().numpy().transpose((1, 2, 0)))
    plt.show()
    y_true = y_true[0]*100
    y_true = torch.cat((y_true, torch.zeros((1, crop_size,crop_size))))
    plt.imshow(y_true.long().numpy().transpose((1, 2, 0)))
    plt.show()
    plt.imshow(pred.long().numpy().transpose((1, 2, 0)))
    plt.show()


