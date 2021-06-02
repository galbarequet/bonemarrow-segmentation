import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters
from skimage.segmentation import flood

BACKGROUND = 0
FAT = 1
CELLS = 2
BONE = 3
BLOCK_SIZE = 6
RED_ARR = np.array([255,0,0]).astype(np.uint8)
GREEN_ARR = np.array([0,255,0]).astype(np.uint8)
BLUE_ARR = np.array([0,0,255]).astype(np.uint8)
WHITE_ARR = np.array([255,255,255]).astype(np.uint8)


def main(image_dir, path_to_save, filename, show_segmentation_bool = False):
    path_to_image = os.path.join(image_dir, filename)
    path_to_segment = os.path.join(path_to_save, filename[:-len("_raw_image.png")]+'_base_seg.npy')
    image = Image.open(path_to_image)
    image_data = np.asarray(image)
    segmented_image = segment_by_color(image_data)
    print("segmented image by color")
    if(show_segmentation_bool):
        show_segmentation(segmented_image, save_bool=True, path_to_save="./seg1.png")
    np.save(path_to_segment, segmented_image)
    segmented_image = np.load(path_to_segment)
    segmented_image = separate_fat_and_background(image_data, segmented_image)
    print("segmented fat and background")
    if(show_segmentation_bool):
        show_segmentation(segmented_image, save_bool=True, path_to_save="./seg2.png")
    np.save(path_to_segment, segmented_image)
    segmented_image = np.load(path_to_segment)
    segmented_image = seperate_bone_and_cells(image_data, segmented_image)
    print("segmented cells and bone")
    np.save(path_to_segment, segmented_image)
    segmented_image = np.load(path_to_segment)
    if(show_segmentation_bool):
        show_segmentation(segmented_image, save_bool=True, path_to_save="./seg3.png")
    # return segmented_image


def segment_by_color(image_data):
    """
        Input: An ndarray of shape (w,h,3) (for each pixel its RGB values)
        Return: An ndarray of shape (w,h), for each pixel its class number:
            Background/Fat = 0
            Bone/Cells = 2
    """
    segmented_image = np.zeros((image_data.shape[0], image_data.shape[1]))
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if np.amax(image_data[x,y]) - np.amin(image_data[x,y]) > 50:
                segmented_image[x,y] = CELLS
            # check if the color is "pink" and if so change the value in segmented_image to 2
    return segmented_image


def separate_fat_and_background(image_data, segmented_image):
    """
        Input: image_data - An ndarray of shape (w,h,3) (for each pixel its RGB values)
               segmented_image - An ndarray of shape (w,h), for each pixel its class number 
        Return: An ndarray of the same shape and meanning when we segment 
                connected components of fat which are too big as background
    """
    image_sobel = filters.sobel(image_data[..., 0])
    flooded_image = flood(image_sobel, (0, 0), tolerance=0.01)
    flooded_image = np.logical_or(flooded_image, flood(image_sobel, (image_sobel.shape[0]-1, 0), tolerance=0.01))
    flooded_image = np.logical_or(flooded_image, flood(image_sobel, (0, image_sobel.shape[1]-1), tolerance=0.01))
    flooded_image = np.logical_or(flooded_image, flood(image_sobel, (image_sobel.shape[0]-1, image_sobel.shape[1]-1), tolerance=0.01))
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if segmented_image[x,y] == BACKGROUND and flooded_image[x,y]:
                segmented_image[x,y] = BACKGROUND
            elif segmented_image[x,y] == BACKGROUND:
                segmented_image[x,y] = FAT
    return segmented_image
    

def seperate_bone_and_cells(image_data, segmented_image):
    bone_mask = np.ndarray(segmented_image.shape, dtype = bool)
    image_sobel = filters.sobel(image_data[..., 1])
    for x in range(0,image_data.shape[0], BLOCK_SIZE):
        for y in range(0,image_data.shape[1], BLOCK_SIZE):
            if(bone_mask[x,y]):
                continue
            # 1. check block is mostly segmeneted as bone/bone-marrow
            block = image_data[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
            if np.count_nonzero(segmented_image[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE] != CELLS) > (BLOCK_SIZE ** 2) / 2:
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

    # # padding the bones for compensation
    # for x in range(image_data.shape[0]):
    #     for y in range(image_data.shape[1]):
    #         if bone_mask[x,y] and segmented_image[x,y] == CELLS:
    #             bone_mask[x - 1: x + 1, y - 1: y + 1] = np.ones(bone_mask[x - 1: x + 1, y - 1: y + 1].shape) * BONE

    # changing to bones
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if segmented_image[x,y] == CELLS and bone_mask[x,y]:
                segmented_image[x,y] = BONE
    
    # connecting bones
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            if segmented_image[x,y] == BONE and segmented_image[min(x+5, image_data.shape[0]-1),y] == BONE:
                for x1 in range(0,5):
                    segmented_image[min(x+x1, image_data.shape[0]-1),y] = BONE
            if segmented_image[x,y] == BONE and segmented_image[x,min(y+5, image_data.shape[1]-1)] == BONE:
                for y1 in range(0,5):
                    segmented_image[x,min(y+y1, image_data.shape[1]-1)] = BONE

    return segmented_image

def show_segmentation(segmented_image, save_bool = False, path_to_save = None, show_bool = True):
    """
        Prints the different segments
    """
    image = np.ndarray((segmented_image.shape[0], segmented_image.shape[1], 3))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if segmented_image[x,y] == BACKGROUND:
                image[x,y] = RED_ARR
            elif segmented_image[x,y] == FAT:
                image[x,y] = BLUE_ARR
            elif segmented_image[x,y] == CELLS:
                image[x,y] = GREEN_ARR
            else:
                image[x,y] = WHITE_ARR
    if save_bool:
        Image.fromarray(image.astype('uint8'), 'RGB').save(path_to_save)
    if show_bool:
        plt.imshow(image)
        plt.show()

def segment_all_images(image_dir, segmentation_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            main(image_dir, segmentation_dir, filename)
            print("finished file: " + filename)
    return

def save_photo_of_segmentation(path_to_array_dir, path_to_image_dir):
    for filename in os.listdir(path_to_array_dir):
        if filename.endswith(".npy"):
            segmented_image = np.load(os.path.join(path_to_array_dir, filename))
            show_segmentation(segmented_image, save_bool=True, path_to_save=os.path.join(path_to_image_dir, filename[:-3]+"png"), show_bool=False)
            print("finished file: " + filename)
    return

def show_segmantation_from_file(path_to_arr):
    """
        Given a path to a segmentation array prints it.
    """
    segmented_image = np.load(path_to_arr)
    show_segmentation(segmented_image)

def dsc(y_pred, y_true):
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def evaluate_baseline(path_to_baseline_seg_arr_dir, path_data_dir):
    dsc_bones = []
    dsc_fat = []
    name_list = os.listdir(os.path.join(path_data_dir, "PS_files"))
    for filename in name_list:
        if filename.endswith(".psd"):
            real_filename = filename[ : -4]
            fat_mask = np.array( Image.open(os.path.join(path_data_dir, "fat/" + real_filename + "_fat.png")) )
            fat_mask = fat_mask / 255
            bones_mask = np.array( Image.open(os.path.join(path_data_dir, "bones/" + real_filename + "_bones.png")) )
            bones_mask = bones_mask / 255
            baseline_seg_arr = np.load( os.path.join(path_to_baseline_seg_arr_dir, real_filename + "_base_seg.npy") )
            fat_base_seg_mask = np.where( baseline_seg_arr == FAT, np.ones(baseline_seg_arr.shape), np.zeros(baseline_seg_arr.shape) )
            bones_base_seg_mask = np.where( baseline_seg_arr == BONE, np.ones(baseline_seg_arr.shape), np.zeros(baseline_seg_arr.shape) )
            dsc_fat.append( dsc(fat_mask, fat_base_seg_mask) )
            dsc_bones.append( dsc(bones_mask, bones_base_seg_mask) )
            # print(filename + " - fat: " + str(dsc_fat[-1]) + " - bones: " + str(dsc_bones[-1]))
        # print("finidhed " + filename)
    max_dsc_fat = np.max(np.array(dsc_fat))
    max_dsc_bones = np.max(np.array(dsc_bones))
    min_dsc_fat = np.min(np.array(dsc_fat))
    min_dsc_bones = np.min(np.array(dsc_bones))
    median_dsc_fat = np.median(np.array(dsc_fat))
    median_dsc_bone = np.median(np.array(dsc_bones))
    mean_dsc_fat = np.mean(np.array(dsc_fat))
    mean_dsc_bone = np.mean(np.array(dsc_bones))
    print("mean dsc for fat = " + str(mean_dsc_fat))
    print("median dsc for fat = " + str(median_dsc_fat))
    print("max dsc for fat = " + str(max_dsc_fat))
    print("min dsc for fat = " + str(min_dsc_fat))
    print("mean dsc for bones = " + str(mean_dsc_bone))
    print("median dsc for bones = " + str(median_dsc_bone))
    print("max dsc for bones = " + str(max_dsc_bones))
    print("min dsc for bones = " + str(min_dsc_bones))

#### Run ####
path_data_dir = "../Real_DATA"
path_image_dir = "../Real_DATA/raw_image"
path_segmentation_dir = "../Baseline_seg"
path_seg_img_dir = "../Baseline_seg/Image"

# segment_all_images(path_image_dir, path_segmentation_dir)
# save_photo_of_segmentation(path_segmentation_dir, path_seg_img_dir)
evaluate_baseline(path_segmentation_dir, path_data_dir)