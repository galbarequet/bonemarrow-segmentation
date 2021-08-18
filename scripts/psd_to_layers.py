import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import numpy as np
from PIL import Image
from psd_tools import PSDImage

def psd_to_layers(psd_dir, save_dir):
    raw_image_dir_path = os.path.join(save_dir, "raw_images")
    background_dir_path = os.path.join(save_dir, "background")
    fat_dir_path = os.path.join(save_dir, "fat")
    bones_dir_path = os.path.join(save_dir, "bones")
    os.makedirs(background_dir_path, exist_ok = True)
    os.makedirs(raw_image_dir_path, exist_ok = True)
    os.makedirs(fat_dir_path, exist_ok = True)
    os.makedirs(bones_dir_path, exist_ok = True)
    for filename in os.listdir(psd_dir):
        try:
            print(filename)
            if filename.endswith(".psd"):
                psd_file = PSDImage.open(os.path.join(psd_dir, filename))
                layer_raw_image = psd_file[0]
                layer_no_background = psd_file[1]
                for layer in psd_file[2:]: # for some reason the layers are switching order from photo to photo
                    if layer.mask is None:
                        layer_bones = layer
                    else:
                        layer_fat = layer

                raw_image = layer_raw_image.topil()
                raw_image_arr = layer_raw_image.numpy()
                raw_image.save(os.path.join(raw_image_dir_path, filename[:-4] + ".png"))

                #creates a mask for the background (1 - not-background, 0 - background)
                no_background_arr = np.zeros((raw_image_arr.shape[0], raw_image_arr.shape[1]))
                bb_no_background = layer_no_background.bbox
                no_background_arr[bb_no_background[1]: bb_no_background[3] , bb_no_background[0] : bb_no_background[2]] = layer_no_background.numpy()[: , : , 3]
                background_arr = np.where(no_background_arr < 0.1, np.zeros(no_background_arr.shape), np.ones(no_background_arr.shape))
                background = Image.fromarray((background_arr * 255).astype(np.uint8))
                background.save(os.path.join(background_dir_path, filename[:-4] + ".png"))

                #creates a mask for the bones (1 - bone, 0 - not bone)
                bones_arr = np.zeros((raw_image_arr.shape[0], raw_image_arr.shape[1]))
                bb_bones = layer_bones.bbox
                bones_arr[bb_bones[1]: bb_bones[3] , bb_bones[0] : bb_bones[2]] = layer_bones.numpy()[: , : , 3]
                bones_arr = np.where(bones_arr < 0.1, np.zeros(bones_arr.shape), np.ones(bones_arr.shape))
                bones = Image.fromarray((bones_arr * 255).astype(np.uint8))
                bones.save(os.path.join(bones_dir_path, filename[:-4] + ".png"))

                #creates a mask for the fat (1 - fat, 0 - not fat)
                fat_arr = np.zeros((raw_image_arr.shape[0], raw_image_arr.shape[1]))
                bb_fat = layer_fat.bbox
                fat_arr[bb_fat[1]: bb_fat[3] , bb_fat[0] : bb_fat[2]] = np.array(layer_fat.mask.topil())
                fat_arr = np.where(fat_arr < 0.1, np.zeros(fat_arr.shape), np.ones(fat_arr.shape))
                fat = Image.fromarray((fat_arr * 255).astype(np.uint8))
                fat.save(os.path.join(fat_dir_path, filename[:-4] + ".png"))
        except:
            print('error {}'.format(filename))
    return


#### Run ####
path_ps_dir = "../psd_samples"
path_save_dir = "../data_samples"

psd_to_layers(path_ps_dir, path_save_dir)
