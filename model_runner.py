import dataset
import events
from hannahmontananet import HannahMontanaNet
import numpy as np
from skimage.io import imsave
import sliding_window
import torch
import utils


class ModelRunner:
    def __init__(self, weights_path, crop_size, step_size):
        # CR: (GB) change device to arg or something
        self._device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

        self._load_model(weights_path)

        self._sliding_window = sliding_window.SlidingWindow(self._net, crop_size, step_size)
        self.progress_event = events.Events()
        self._sliding_window.progress_event.on_change += self._trigger_progress_event

    def run_segmentation(self, image):
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0)

        with torch.set_grad_enabled(False):
            x = image.to(self._device)
            y_pred = self._sliding_window.predict_image(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            utils.remove_lowest_confidence(y_pred_np)
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

    def _trigger_progress_event(self, value):
        self.progress_event.on_change(value)
