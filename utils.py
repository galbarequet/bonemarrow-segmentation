import numpy as np
import torch
import torch.nn.functional as F
from skimage.exposure import rescale_intensity
from skimage.transform import resize


def dsc(y_pred, y_true):
    eps = 1e-3
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    bone_dsc = (2*np.sum(y_pred[0] * y_true[0]) + eps) / ((np.sum(y_true[0]) + np.sum(y_pred[0])) + eps)
    fat_dsc = (2*np.sum(y_pred[1] * y_true[1])+ eps) / ((np.sum(y_true[1]) + np.sum(y_pred[1])) + eps)
    return bone_dsc, fat_dsc


def crop_sample(x):
    volume, mask = x
    volume[volume < np.max(volume) * 0.1] = 0
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_to_multiple(x, divisor):
    volume, mask = x
    a = volume.shape[0]
    b = volume.shape[1]

    new_a = a + (divisor - (a % divisor))
    new_b = b + (divisor - (b % divisor))
    diff_a = new_a - a
    diff_b = new_b - b
    padding = ((0, diff_a), (0, diff_b), (0,0))
    volume = np.pad(volume, padding, mode='constant', constant_values=255)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)

    return volume, mask

def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask


def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask


def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume


def log_images(x, y_true, y_pred, channel=1):
    images = []
    x_np = x[:, channel].cpu().numpy()
    y_true_np = y_true[:, 0].cpu().numpy()
    y_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(x_np.shape[0]):
        image = gray2rgb(np.squeeze(x_np[i]))
        image = outline(image, y_pred_np[i], color=[255, 0, 0])
        image = outline(image, y_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images


def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image

def pad_to_multiple_tf(x, divisor):
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
    padding = (0,diff_b, 0,diff_a)
    return F.pad(input=x, pad=padding, mode='constant', value=255)

def pred_image_crop(image, network, crop_size, step_size = None):
    """
        image - tensor of the image to segment
        network - the segmenting network
        crop_size - the size of the squres on which to segment
        step_size - (default - crop_size) the step size taken by the sliding window

        return - a tensor of the segmented image
    """
    if step_size == None:
        step_size = crop_size

    pred_device = "cpu"
    if image.is_cuda:
        pred_device = image.get_device()
    data_type = image.dtype
    
    h = image.shape[2]
    w = image.shape[3]
    y_pred = torch.zeros(size=(image.shape[0], network.out_channels, h, w), dtype=data_type, device=pred_device)
    weights = torch.zeros(size=(image.shape[0], network.out_channels, h, w), dtype=data_type, device=pred_device)
    for start_x in range(0, h, step_size):
        for start_y in range(0, w, step_size):
            end_x = min(start_x + crop_size, h)
            end_y = min(start_y + crop_size, w)
            cropped_sample = image[:, :, start_x : end_x, start_y : end_y]
            cropped_sample = pad_to_multiple_tf(cropped_sample, 256)
            cropped_y_pred = network(cropped_sample)
            y_pred[:, :, start_x : end_x, start_y : end_y] += cropped_y_pred[:, :, : end_x - start_x, : end_y - start_y]
            weights[:, :, start_x : end_x, start_y : end_y] += torch.ones(size=(image.shape[0], network.out_channels, end_x - start_x, end_y - start_y), dtype=data_type, device=pred_device)
            # print("{}-{}".format(start_x, start_y))
    y_pred = torch.div(y_pred, weights)
    return y_pred

def create_seg_image(seg):
    seg = seg*100
    seg = np.concatenate((seg, np.zeros((1, seg.shape[1], seg.shape[2]))))
    seg = seg.transpose((1, 2, 0))
    return seg.astype(np.uint8)
