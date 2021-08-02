import numpy as np


def remove_lowest_confidence(y_pred):
    """
    Zeros the channel with lower confidence in prediction. Use this if you want to have an exclusive label per pixel
    """
    y_pred[:, 0, y_pred[0, 0] < y_pred[0, 1]] = 0
    y_pred[:, 1, y_pred[0, 0] >= y_pred[0, 1]] = 0


def dsc(y_pred, y_true, eps=1e-3):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    bone_dsc = (2*np.sum(y_pred[0] * y_true[0]) + eps) / ((np.sum(y_true[0]) + np.sum(y_pred[0])) + eps)
    fat_dsc = (2*np.sum(y_pred[1] * y_true[1]) + eps) / ((np.sum(y_true[1]) + np.sum(y_pred[1])) + eps)
    return bone_dsc, fat_dsc


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image


def create_seg_image(seg):
    seg = seg*100
    seg = np.concatenate((seg, np.zeros((1, seg.shape[1], seg.shape[2]))))
    seg = seg.transpose((1, 2, 0))
    return seg.astype(np.uint8)


def create_error_image(pred_layer, true_layer):
    diff = pred_layer - true_layer
    error = np.stack((diff, -1*diff, np.zeros_like(pred_layer)))

    error = np.maximum(error, 0) * 255
    error = error.transpose((1, 2, 0))

    return error.astype(np.uint8)

