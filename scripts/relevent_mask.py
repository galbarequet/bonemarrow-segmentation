import os

import numpy as np
from skimage.io import imread
from PIL import Image
from skimage import color

edge_size = 256
step = 10

image_dir = r'data_samples'
raw_image_dir = 'fixed images'
background_dir = 'background'
masks_dir = 'masks'


raw_image_full_path = os.path.join(image_dir, raw_image_dir)
back_image_full_path = os.path.join(image_dir, background_dir)
masks_image_full_path = os.path.join(image_dir, masks_dir)
for filename in os.listdir(raw_image_full_path):
    raw_img = np.array(color.rgb2gray(imread(os.path.join(raw_image_full_path, filename))))
    back_img = np.array(imread(os.path.join(back_image_full_path, filename)))
    base_color = 0.9019756862745099
    raw_img = np.pad(raw_img, ((edge_size, edge_size), (edge_size, edge_size)), constant_values=1)
    mask = np.zeros((raw_img.shape[0], raw_img.shape[1]), dtype=np.uint8)

    for i in range(0, raw_img.shape[0] - edge_size, step):
        for j in range(0, raw_img.shape[1] - edge_size, step):
            avg_pixel = np.sum(raw_img[i:i + edge_size, j:j + edge_size]) / edge_size ** 2
            if avg_pixel < base_color * 0.99:
                mask[i: i + step, j: j+step] = 255

    mask = mask[edge_size: mask.shape[0] - edge_size, edge_size: mask.shape[1] - edge_size]
    im = Image.fromarray(mask)
    im.convert('L')
    im.save(os.path.join(masks_image_full_path, filename))
    print('saved')
