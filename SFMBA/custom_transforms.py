import torchvision
import random
import numpy as np
from skimage.transform import resize

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for process in self.transforms:
            images, intrinsics = process(images, intrinsics)

        return images, intrinsics


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, images, intrinsics):
        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = self.width / in_w, self.height / in_h

        output_intrinsics = np.copy(intrinsics)
        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        resized_images = [resize(img, (self.height, self.width)) for img in images]

        return resized_images, output_intrinsics


class Normalize(object):
    def __init__(self, mean, std):
        self.normalizer = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, images, intrinsics):
        output_images = [self.normalizer(img) for img in images]
        return output_images, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix
    to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, images, intrinsics):
        output_images = [self.to_tensor(img).float() for img in images]
        return output_images, intrinsics


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None

        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(img)) for img in images]
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_intrinsics = intrinsics

        return output_images, output_intrinsics


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [resize(img, (scaled_h, scaled_w)) for img in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [img[offset_y:offset_y + in_h,
                          offset_x:offset_x + in_w] for img in scaled_images]

        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        return cropped_images, output_intrinsics


class RandomColorJitter(object):
    """Randomly jitter the color of RGB image"""

    def __call__(self, images, intrinsics):
        if random.random() > 0.5:
            color_augmentation = torchvision.transforms.ColorJitter(
                brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)
            )
            output_images = [color_augmentation(img) for img in images]
        else:
            output_images = images

        return output_images, intrinsics
