import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset


class BoneMarrowDataset(Dataset):
    """Bone Marrow dataset for fat and bones segmentation"""

    in_channels = 3
    out_channels = 4

    @staticmethod
    def create_mask(bone_layer, fat_layer, tissue_layer, fat_overrides_bone):
        bone_layer = (bone_layer/255).astype('uint8')
        fat_layer = (fat_layer/255).astype('uint8')
        tissue_layer = (tissue_layer / 255).astype('uint8')

        if fat_overrides_bone:
            bone_layer[bone_layer == fat_layer] = 0
        else:
            fat_layer[bone_layer == fat_layer] = 0

        tissue_layer[tissue_layer == fat_layer] = 0
        tissue_layer[tissue_layer == bone_layer] = 0

        mask = bone_layer + 2 * fat_layer + 3*tissue_layer

        mask = np.expand_dims(mask, axis=2)
        #mask = np.stack([bone_layer, fat_layer])
        #mask = mask.transpose(1, 2, 0)
        return mask

    @staticmethod
    def load_dataset(image_dir, fat_overrides_bone):
        data = []
        labels = []
        names = []
        crop_masks = []
        background_image_dir = 'background'
        raw_image_dir = 'raw_images'
        bone_layer_dir = 'bones'
        fat_layer_dir = 'fat'
        mask_dir = 'masks'
        raw_image_full_path = os.path.join(image_dir, raw_image_dir)
        for i, filename in enumerate(os.listdir(raw_image_full_path)):
            raw_img = np.array(imread(os.path.join(raw_image_full_path, filename)))

            bone_layer = np.array(imread(os.path.join(os.path.join(image_dir, bone_layer_dir), filename), as_gray=True))

            fat_layer = np.array(imread(os.path.join(os.path.join(image_dir, fat_layer_dir), filename), as_gray=True))

            tissue_layer = np.array(imread(os.path.join(os.path.join(image_dir, background_image_dir), filename), as_gray=True))

            crop_mask = np.array(imread(os.path.join(os.path.join(image_dir, mask_dir), filename), as_gray=True))

            names.append(filename)
            data.append(raw_img)
            crop_masks.append(crop_mask)
            labels.append(BoneMarrowDataset.create_mask(bone_layer, fat_layer, tissue_layer, fat_overrides_bone))

        return data, labels, names, crop_masks

    def __init__(
        self,
        images_dir,
        transform=None,
        subset="train",
        random_sampling=True,
        validation_cases=7,
        seed=42,
        fat_overrides_bone=True
    ):
        assert subset in ["all", "train", "validation"]



        # read images
        self.images, self.labels, self.names, self.crop_masks = self.load_dataset(images_dir, fat_overrides_bone)

        # select cases to subset
        # TODO: This random shit is way to wack please fix this
        # Note: in training we use random crop in matching size, and in validation we use sliding window
        #       so no need to add padding to images in either case
        if not subset == "all":
            random.seed(seed)
            indices = [i for i in range(len(self.labels))]
            validation_indices = indices
            if validation_cases is not None:
                validation_indices = random.sample(indices, k=validation_cases)
            if subset != "validation":
                train_indices = sorted(
                    list(set(indices).difference(validation_indices))
                )
                self.images = [self.images[i] for i in train_indices]
                self.labels = [self.labels[i] for i in train_indices]
                self.names = [self.names[i] for i in train_indices]
                self.crop_masks = [self.crop_masks[i] for i in train_indices]
            else:
                self.images = [self.images[i] for i in validation_indices]
                self.labels = [self.labels[i] for i in validation_indices]
                self.names = [self.names[i] for i in validation_indices]
                self.crop_masks = [self.crop_masks[i] for i in validation_indices]

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        crop_mask = self.crop_masks[idx]

        # Note: in validation self.random_sampling ia false, so we can use idx in names attribute properly
        if self.random_sampling:
            idx = np.random.randint(len(self.images))
            image = self.images[idx]
            label = self.labels[idx]
            crop_mask = self.crop_masks[idx]

        if self.transform is not None:
            image, label = self.transform((image, label, crop_mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        label_tensor = torch.from_numpy(label.astype(np.int)).long()
        label_tensor = torch.squeeze(label_tensor)

        # return tensors
        return image_tensor, label_tensor
