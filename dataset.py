import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume, pad_to_multiple


class BoneMarrowDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 2


    def create_mask(self, bone_layer, fat_layer):
        bone_layer = (bone_layer/255).astype('uint8')
        fat_layer = (fat_layer/255).astype('uint8')
        #background_layer = np.zeros_like(bone_layer, dtype='uint8')

        #background_layer = background_layer + bone_layer + fat_layer
        #background_layer[background_layer != 0] = 1
        #background_layer = 1 - background_layer
        #background_layer = background_layer.astype('uint8')

        mask = np.stack([bone_layer, fat_layer])
        mask = mask.transpose(1, 2, 0)
        return mask


    def load_dataset(self, image_dir):
        data = []
        masks = []
        raw_image_dir = 'raw_image'
        bone_layer_dir = 'bones'
        fat_layer_dir = 'fat'
        raw_image_full_path = os.path.join(image_dir, raw_image_dir)
        for i, filename in enumerate(os.listdir(raw_image_full_path)):
            raw_img = np.array(imread(os.path.join(raw_image_full_path, filename)))

            bone_layer = np.array(imread(os.path.join(os.path.join(image_dir, bone_layer_dir), filename), as_gray=True))

            fat_layer = np.array(imread(os.path.join(os.path.join(image_dir, fat_layer_dir), filename), as_gray=True))

            data.append(raw_img)
            masks.append(self.create_mask(bone_layer, fat_layer))

        return data, masks



    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=4,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        self.images, self.masks = self.load_dataset(images_dir)


        # select cases to subset
        #TODO: This random shit is way to wack please fix this
        if not subset == "all":
            random.seed(seed)
            indices = [i for i in range(len(self.masks))]
            validation_indices = random.sample(indices, k=validation_cases)
            validation_images = [self.images[i] for i in validation_indices]
            validation_masks = [self.masks[i] for i in validation_indices]
            if subset == "validation":
                #self.images, self.mask = zip(*[pad_to_multiple((self.images[i], self.masks[i]), 32) for i in validation_indices])
                self.images = validation_images
                self.masks = validation_masks
            else:
                train_indices = sorted(
                    list(set(indices).difference(validation_indices))
                )
                self.images = [self.images[i] for i in train_indices]
                self.masks = [self.masks[i] for i in train_indices]


        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.random_sampling:
            idx = np.random.randint(len(self.images))
            image = self.images[idx]
            mask = self.masks[idx]


        if self.transform is not None:
            image, mask = self.transform([image, mask])
            #mask_tensor = self.transform(mask_tensor)

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor
