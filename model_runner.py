from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

import cv2
import dataset
from hannahmontananet import HannahMontanaNet
import matplotlib
import numpy as np
from PIL import Image, ImageOps
import torch
import utils
matplotlib.use('Agg')
from skimage.io import imsave

# CR: (GB) perhaps should remove this? or maybe change values and see what they mean...
INPUT_DIM = (256, 256)
OUTPUT_DIM = (64, 64)


class ModelRunner:
    def __init__(self, weights_path, crop_size, step_size):
        # CR: (GB) change device to arg or something
        self._device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

        self._load_model(weights_path)

        self.crop_size = crop_size
        self.step_size = step_size

    def run_segmentation(self, image):
        padding = utils.get_padding_by_multiple(image, 32)
        image = np.pad(image, padding, mode='constant', constant_values=255)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0)

        with torch.set_grad_enabled(False):
            x = image.to(self._device)
            y_pred = utils.pred_image_crop(x, self._net, self.crop_size, step_size=self.step_size)
            y_pred_np = y_pred.detach().cpu().numpy()
            y_pred_np = np.round(y_pred_np).astype(np.int)

            segmented_image = utils.create_seg_image(y_pred_np[0])
            imsave(r'D:\Gali\university\tau\ML workshop\dataset\dataset_test\app\pred.png', segmented_image)
            return segmented_image

    def _load_model(self, weights_path):
        with torch.set_grad_enabled(False):
            self._net = HannahMontanaNet(out_channels=dataset.BoneMarrowDataset.out_channels)
            state_dict = torch.load(weights_path, map_location=self._device)
            self._net.load_state_dict(state_dict)
            self._net.eval()
            self._net.to(self._device)
