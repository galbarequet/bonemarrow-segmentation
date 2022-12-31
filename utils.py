from bonemarrow_label import BoneMarrowLabel
import matplotlib.pyplot as plt
import numpy as np


def calculate_density(target_mask, mass_type=BoneMarrowLabel.BONE, eps=1e-3):
    bone_pixels = target_mask == mass_type
    tissue_pixels = target_mask != BoneMarrowLabel.BACKGROUND

    bone_density = (np.sum(bone_pixels) + eps) / (np.sum(tissue_pixels) + eps)
    return bone_density


def calculate_bonemarrow_density_error(y_pred, y_true):
    pred_bone_density = calculate_density(y_pred)
    true_bone_density = calculate_density(y_true)
    return np.abs(pred_bone_density - true_bone_density) / true_bone_density


def dsc(y_pred, y_true, eps=1e-3, categories=BoneMarrowLabel.TOTAL):
    dscs = []

    for i in range(categories):
        pred_category = y_pred == i
        true_category = y_true == i
        dsc = (2 * np.sum(pred_category * true_category) + eps) / ((np.sum(pred_category) + np.sum(true_category)) + eps)
        dscs.append(dsc)

    return tuple(dscs)


def create_seg_image(seg):
    bone_seg = seg == BoneMarrowLabel.BONE
    fat_seg = seg == BoneMarrowLabel.FAT
    tissue_seg = seg == BoneMarrowLabel.OTHER
    seg = np.stack((bone_seg, fat_seg, tissue_seg), axis=2)
    return seg.astype(np.uint8) * 100


def create_error_image(pred, true, category):
    false_positive = (pred == category) * (true != category)
    false_negative = (pred != category) * (true == category)

    error = np.stack((false_positive, false_negative, np.zeros_like(pred)), axis=2) * 100

    return error.astype(np.uint8)


def create_density_figure(prediction):
    labels_values = [BoneMarrowLabel.BONE, BoneMarrowLabel.FAT, BoneMarrowLabel.OTHER]
    densities = [100 * calculate_density(prediction, label) for label in labels_values]

    fig, ax = plt.subplots()
    labels = ['Bone Mass', 'Fat Mass', 'Other Tissue Mass']
    explode = [0.1, 0, 0]
    colors = ['#640000', '#006400', '#000064']
    _, _, autotexts = ax.pie(densities, labels=labels, autopct='%1.2f%%', startangle=90, explode=explode, colors=colors)
    for autotext in autotexts:
        autotext.set_color('white')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    densities_mapping = {
        label: density
        for label, density in zip(labels, densities)
    }

    return fig, ax, densities_mapping
