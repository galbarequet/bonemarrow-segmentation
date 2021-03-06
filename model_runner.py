import dataset
import events
from hannahmontananet import HannahMontanaNet
import numpy as np
import sliding_window
import torch


class ModelRunner:
    def __init__(self, weights_path, crop_size, step_size):
        self._device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

        self._load_model(weights_path)

        self._sliding_window = sliding_window.SlidingWindow(self._net, crop_size, step_size)
        self.progress_event = events.Events()
        self._sliding_window.progress_event.on_change += self._trigger_progress_event

    def predict(self, image):
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0)

        with torch.set_grad_enabled(False):
            x = image.to(self._device)
            y_pred = self._sliding_window.predict_image(x)
            y_pred_np = y_pred.detach().cpu().numpy()[0, ...]
            return np.argmax(y_pred_np, axis=0)

    def _load_model(self, weights_path):
        with torch.set_grad_enabled(False):
            self._net = HannahMontanaNet(out_channels=dataset.BoneMarrowDataset.out_channels)
            state_dict = torch.load(weights_path, map_location=self._device)
            self._net.load_state_dict(state_dict)
            self._net.eval()
            self._net.to(self._device)

    def _trigger_progress_event(self, value):
        self.progress_event.on_change(value)
