import numpy as np


def calculate_bonemarrow_density_error(y_pred, y_true, eps=1e-3):
    pred_bone_pixels = y_pred == 1
    pred_tissue_pixels = y_pred != 0

    pred_bone_density = (np.sum(pred_bone_pixels) + eps) / (np.sum(pred_tissue_pixels) + eps)

    true_bone_pixels = y_true == 1
    true_tissue_pixels = y_true != 0

    true_bone_density = (np.sum(true_bone_pixels) + eps) / (np.sum(true_tissue_pixels) + eps)

    error = np.abs(pred_bone_density - true_bone_density) / true_bone_density

    return error


def remove_lowest_confidence(y_pred):
    """
    Zeros the channel with lower confidence in prediction. Use this if you want to have an exclusive label per pixel
    """
    y_pred[:, 0, y_pred[0, 0] < y_pred[0, 1]] = 0
    y_pred[:, 1, y_pred[0, 0] >= y_pred[0, 1]] = 0


def dsc(y_pred, y_true, eps=1e-3, categories=4):
    dscs = []

    for i in range(categories):
        pred_category = y_pred == i
        true_category = y_true == i
        dsc = (2 * np.sum(pred_category * true_category) + eps) / ((np.sum(pred_category) + np.sum(true_category)) + eps)
        dscs.append(dsc)

    return tuple(dscs)


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image


def create_seg_image(seg):
    bone_seg = seg == 1
    fat_seg = seg == 2
    tissue_seg = seg == 3
    seg = np.stack((bone_seg, fat_seg, tissue_seg), axis=2)
    #seg = seg.transpose((1, 2, 0))
    return seg.astype(np.uint8) * 100


def create_error_image(pred, true, category):
    false_positive = (pred == category) * (true != category)
    false_negative = (pred != category) * (true == category)

    error = np.stack((false_positive, false_negative, np.zeros_like(pred)), axis=2) * 100

    return error.astype(np.uint8)

