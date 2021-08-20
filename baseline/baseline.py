import os
import numpy as np
from skimage import filters
from skimage.segmentation import flood
import sys
sys.path.append(os.getcwd())
from bonemarrow_label import BoneMarrowLabel


BLOCK_SIZE = 6


def segment_image(image_data, print_progress = False):
    """
        Given an image array returns the segmented image
    """
    segmented_image = segment_by_color(image_data)
    if print_progress:
        print("segmented image by color")
    segmented_image = separate_fat_and_background(image_data, segmented_image)
    if print_progress:
        print("segmented fat and background")
    segmented_image = seperate_bone_and_other(image_data, segmented_image)
    if print_progress:
        print("segmented other and bone")
    return segmented_image


def segment_by_color(image_data):
    """
        Input: An ndarray of shape (w,h,3) (for each pixel its RGB values)
        Return: An ndarray of shape (w,h), for each pixel its class number:
            background and fat are segmented as BACKGROUND
            bone and other are segmenteda as OTHER
    """
    # check if the color is "pink" and if so change the value in segmented_image to OTHER
    return ((np.amax(image_data, axis=-1) - np.amin(image_data, axis=-1)) > 50) * BoneMarrowLabel.OTHER


def separate_fat_and_background(image_data, segmented_image):
    """
        Input: image_data - An ndarray of shape (w,h,3) (for each pixel its RGB values)
               segmented_image - An ndarray of shape (w,h), for each pixel its class number 
        Return: An ndarray of the same shape and meanning when we segment 
                connected components of fat which are too big as background
    """
    image_sobel = filters.sobel(image_data[..., 0])
    flooded_image = np.zeros(image_sobel.shape)
    for x in [0, image_sobel.shape[0]-1]:
        for y in [0, image_sobel.shape[1]-1]:
            flooded_image = np.logical_or(flooded_image, flood(image_sobel, (x, y), tolerance=0.01))
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if (segmented_image[x, y] == BoneMarrowLabel.BACKGROUND) and (not flooded_image[x, y]):
                segmented_image[x, y] = BoneMarrowLabel.FAT
    return segmented_image
    

def seperate_bone_and_other(image_data, segmented_image):
    bone_mask = np.ndarray(segmented_image.shape, dtype = bool)
    image_sobel = filters.sobel(image_data[..., 1])
    for x in range(0,image_data.shape[0], BLOCK_SIZE):
        for y in range(0,image_data.shape[1], BLOCK_SIZE):
            if(bone_mask[x,y]):
                continue
            # 1. check block is mostly segmeneted as bone/bone-marrow
            block = image_data[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
            if np.count_nonzero(segmented_image[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE] != BoneMarrowLabel.OTHER) > (BLOCK_SIZE ** 2) / 2:
                continue
            
            # 2. check block is homogeneous (pink) to be bone
            is_homogeneous_block = True
            for i in range(3):
                values = np.take(block, [i], 2).flatten()
                values.sort()
                if values[values.shape[0]-3] - values[2] > 30:
                    is_homogeneous_block = False
                    break

            if is_homogeneous_block:
                flooded_image = flood(image_sobel, (x, y), tolerance=0.025)
                bone_mask = np.logical_or(flooded_image, bone_mask)
    
    # changing to bones
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if segmented_image[x,y] == BoneMarrowLabel.OTHER and bone_mask[x,y]:
                segmented_image[x,y] = BoneMarrowLabel.BONE
    
    # connecting bones
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if segmented_image[x,y] == BoneMarrowLabel.BONE and segmented_image[min(x+5, image_data.shape[0]-1),y] == BoneMarrowLabel.BONE:
                for x1 in range(0,5):
                    segmented_image[min(x+x1, image_data.shape[0]-1),y] = BoneMarrowLabel.BONE
            if segmented_image[x,y] == BoneMarrowLabel.BONE and segmented_image[x,min(y+5, image_data.shape[1]-1)] == BoneMarrowLabel.BONE:
                for y1 in range(0,5):
                    segmented_image[x,min(y+y1, image_data.shape[1]-1)] = BoneMarrowLabel.BONE

    return segmented_image
