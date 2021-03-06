import events
import numpy as np
import torch
import torch.nn.functional as F


class SlidingWindow:
    def __init__(self, network, crop_size, step_size, window_edge_percentage=0.1, window_edge_weight=0.75):
        self._network = network
        self._crop_size = crop_size
        self._step_size = step_size
        self.window_edge_percentage_halved = window_edge_percentage / 2
        self.weight_to_add_to_center = 1 / window_edge_weight - 1
        self.progress_event = events.Events()

    def predict_image(self, image):
        """
            image - tensor of the image to segment
            network - the segmenting network
            crop_size - the size of the squares on which to segment
            step_size - (default - crop_size) the step size taken by the sliding window

            return - a tensor of the segmented image
        """
        device = "cpu"
        if image.is_cuda:
            device = image.get_device()
        data_type = image.dtype

        h = image.shape[2]
        w = image.shape[3]
        y_pred = torch.zeros(size=(image.shape[0], self._network.out_channels, h, w), dtype=data_type, device=device)
        weights = torch.zeros(size=(image.shape[0], self._network.out_channels, h, w), dtype=data_type, device=device)
        for start_x in range(0, h, self._step_size):
            for start_y in range(0, w, self._step_size):
                end_x = min(start_x + self._crop_size, h)
                end_y = min(start_y + self._crop_size, w)

                window_length = end_x - start_x
                window_height = end_y - start_y

                x_center_start = int(np.floor(start_x + window_length * self.window_edge_percentage_halved))
                x_center_end = int(np.ceil(end_x - window_length * self.window_edge_percentage_halved))
                y_center_start = int(np.floor(start_y + window_height * self.window_edge_percentage_halved))
                y_center_end = int(np.ceil(end_y - window_height * self.window_edge_percentage_halved))

                cropped_sample = image[:, :, start_x: end_x, start_y: end_y]
                cropped_sample = self.pad_tensor_to_multiple(cropped_sample, 256)
                cropped_y_pred = self._network(cropped_sample)

                y_pred[:, :, start_x: end_x, start_y: end_y] += cropped_y_pred[:, :, : end_x - start_x,
                                                                               : end_y - start_y]

                y_pred[:, :, x_center_start: x_center_end, y_center_start: y_center_end] += cropped_y_pred[:, :, x_center_start - start_x: x_center_end - start_x,
                                                                                            y_center_start - start_y: y_center_end - start_y] * self.weight_to_add_to_center

                weights[:, :, start_x: end_x, start_y: end_y] += torch.ones(
                    size=(image.shape[0], self._network.out_channels, end_x - start_x, end_y - start_y),
                    dtype=data_type, device=device)

                weights[:, :, x_center_start: x_center_end, y_center_start: y_center_end] += torch.ones(
                    size=(image.shape[0], self._network.out_channels, x_center_end - x_center_start, y_center_end - y_center_start),
                    dtype=data_type, device=device) * self.weight_to_add_to_center
            self.progress_event.on_change(int(100 * start_x / h))
        self.progress_event.on_change(100)

        # Note: we divide by weights to normalize the prediction in case there was overlapping in the sliding window
        #       which caused summation of multiple window predictions over same pixel.
        y_pred = torch.div(y_pred, weights)
        return y_pred

    @staticmethod
    def pad_tensor_to_multiple(x, divisor):
        a = x.shape[2]
        b = x.shape[3]

        if a % divisor != 0:
            new_a = a + (divisor - (a % divisor))
        else:
            new_a = a
        if b % divisor != 0:
            new_b = b + (divisor - (b % divisor))
        else:
            new_b = b
        diff_a = new_a - a
        diff_b = new_b - b
        padding = (0, diff_b, 0, diff_a)

        return F.pad(input=x, pad=padding, mode='constant', value=255)
