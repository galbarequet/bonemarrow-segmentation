import numpy as np
import imgaug.augmenters as iaa #TODO maybe change everything to imgaug
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose
from PIL import Image, ImageEnhance


def transforms(scale=None, angle=None, flip_prob=None, crop=None, color_apply=None, elastic_apply = None):
    transform_list = []
    if crop is not None:
        transform_list.append(RandomCrop(crop))
    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))
    if color_apply is not None:
        transform_list.append(RandomColorTransform(color_apply))
    if elastic_apply is not None:
        transform_list.append(RandomElasticTransform(elastic_apply))

    return Compose(transform_list)


class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(mask, angle, resize=False, order=0, preserve_range=True, mode="constant")
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask

class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, mask = sample
        height = image.shape[0]
        width = image.shape[1]
        start_height = np.random.randint(0, height - self.size)
        start_width = np.random.randint(0, width - self.size)
        image = image[start_height:start_height + self.size, start_width:start_width + self.size, :]
        mask = mask[start_height:start_height + self.size, start_width:start_width + self.size, :]

        return image, mask

# Might need to be redone as it sems to have a negative affect on the training process
class RandomColorTransform(object):

    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() < self.p:
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            if np.random.rand() < 0.75:
                factor = np.random.rand() + 0.5
                image_pil = ImageEnhance.Brightness(image_pil).enhance(factor)
            if np.random.rand() < 0.75:
                factor = np.random.rand() + 0.5
                image_pil = ImageEnhance.Contrast(image_pil).enhance(factor)
            if np.random.rand() < 0.75:
                factor = np.random.rand() + 0.5
                image_pil = ImageEnhance.Color(image_pil).enhance(factor)
            if np.random.rand() < 0.75:
                colors = []
                factors = []
                colors.append(Image.new('RGB', size=image_pil.size, color=(255,0, 0)))
                colors.append(Image.new('RGB', size=image_pil.size, color=(0, 255, 0)))
                colors.append(Image.new('RGB', size=image_pil.size, color=(0, 0, 255)))
                factors.append(np.random.rand() * 0.2)
                factors.append(np.random.rand() * 0.2)
                factors.append(np.random.rand() * 0.2)
                index = np.random.randint(0, 3)
                for i in range(3):
                    image_pil = Image.blend(image_pil, colors[(i+index)%3], factors[(i+index)%3])


            image = np.array(image_pil)
        
        return image, mask


class RandomElasticTransform(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        # parameters which seem good
        alpha = 100
        sigma = 10

        if np.random.rand() < self.p:
            shape = image.shape[:2] 
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

            for i in range(image.shape[2]):
                image[..., i] = map_coordinates(image[..., i], indices, order=1).reshape(shape)
            for i in range(mask.shape[2]):
                mask[..., i] = map_coordinates(mask[..., i], indices, order=0).reshape(shape)

        return image, mask
