import random
from typing import Tuple, Union

import timm.data.transforms
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from timm.data.transforms import CenterCropOrPad
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

try:
    import albumentations as alb
    from albumentations.pytorch import ToTensorV2
    has_albumentations = True
except ImportError:
    has_albumentations = False

try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False


def create_transforms(
        name,
        image_size,
        training=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        interpolation: str = 'bicubic',
        crop_margin: bool = False,
        align_long_axis: bool = False,
        fill=255,
):
    # FIXME design a config class to cover coarse and fine-grained aug options
    basic_args = dict(
        training=training,
        image_mean=image_mean,
        image_std=image_std
    )
    adv_args = dict(
        interpolation=interpolation,
        crop_margin=crop_margin,
        align_long_axis=align_long_axis,
        fill=fill,
    )
    if name == 'better':
        return better_transforms(image_size, **basic_args, **adv_args)
    elif name == 'nougat':
        return nougat_transforms(image_size, **basic_args, **adv_args)
    else:
        return legacy_transforms(image_size, **basic_args)


def legacy_transforms(
        image_size,
        image_mean,
        image_std,
        training=False,
):
    # super basic and not so good initial transform for PoC runs
    pp = transforms.Compose([
        transforms.Resize(
            image_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_mean,
            std=image_std,
        )
    ])
    return pp


def better_transforms(
        image_size,
        training=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        interpolation='bicubic',
        crop_margin=False,
        align_long_axis=False,
        fill=255,
):
    # an improved torchvision + custom op transforms (no albumentations)
    interpolation_mode = timm.data.transforms.str_to_interp_mode(interpolation)

    pp = []
    if crop_margin:
        assert has_cv2, 'CV2 needed to use crop margin.'
        pp += [CropMargin()]
    if align_long_axis:
        pp += [AlignLongAxis(image_size, interpolation=interpolation_mode)]

    if training:
        pp += [
            ResizeKeepRatio(
                image_size,
                longest=1,
                interpolation=interpolation,
                random_scale_prob=.05,
                random_scale_range=(0.85, 1.04),
                random_aspect_prob=.05,
                random_aspect_range=(0.9, 1.11),
            ),
            transforms.RandomApply([
                Bitmap()
                ],
                p=.05
            ),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    shear=(0, 3., -3, 0),
                    interpolation=interpolation_mode,
                    fill=fill,
                )],
                p=.05,
            ),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0, 0.04),
                    interpolation=interpolation_mode,
                    fill=fill,
                )],
                p=.05,
            ),
            transforms.RandomApply([
                transforms.ElasticTransform(
                    alpha=50.,
                    sigma=120 * 0.1,
                    interpolation=interpolation_mode,
                    fill=fill
                )],
                p=.05,
            ),
            transforms.RandomApply([
                transforms.ColorJitter(0.1, 0.1)],
                p=.05,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 0.5))],
                p=.05,
            ),
            RandomPad(image_size, fill=fill),
            transforms.CenterCrop(image_size),
        ]
    else:
        pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=interpolation),
            CenterCropOrPad(image_size, fill=fill),
        ]

    pp += [
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ]

    return transforms.Compose(pp)


def nougat_transforms(
        image_size,
        training=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
        align_long_axis=False,
        interpolation='bicubic',
        fill=255,
        crop_margin=False,
):
    assert has_albumentations, 'Albumentations and CV2 needed to use nougat transforms.'

    # albumentations + custom opencv transforms from nougat
    if interpolation == 'bilinear':
        interpolation_mode = 1
    else:
        interpolation_mode = 2  # bicubic

    tv_pp = [transforms.Grayscale()]
    alb_pp = []
    if crop_margin:
        tv_pp += [CropMargin()]
    if align_long_axis:
        tv_pp += [AlignLongAxis(image_size)]

    if training:
        tv_pp += [
            # this should be equivalent to initial resize & pad in Donut prepare_input()
            ResizeKeepRatio(image_size, longest=1, interpolation=interpolation),
            RandomPad(image_size, fill=fill),
        ]
        alb_pp += [
            BitmapAlb(p=0.05),
            alb.OneOf([ErosionAlb((2, 3)), DilationAlb((2, 3))], p=0.02),
            alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.03),
            alb.ShiftScaleRotate(
                shift_limit_x=(0, 0.04),
                shift_limit_y=(0, 0.03),
                scale_limit=(-0.15, 0.03),
                rotate_limit=2,
                border_mode=0,
                interpolation=interpolation_mode,
                value=fill,
                p=0.03,
            ),
            alb.GridDistortion(
                distort_limit=0.05,
                border_mode=0,
                interpolation=interpolation_mode,
                value=fill,
                p=0.04,
            ),
            alb.Compose(
                [
                    alb.Affine(
                        translate_px=(0, 5), always_apply=True, cval=(255, 255, 255)
                    ),
                    alb.ElasticTransform(
                        p=1,
                        alpha=50,
                        sigma=120 * 0.1,
                        alpha_affine=120 * 0.01,
                        border_mode=0,
                        value=fill,
                    ),
                ],
                p=0.04,
            ),
            alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.03),
            alb.ImageCompression(95, p=0.07),
            alb.GaussNoise(20, p=0.08),
            alb.GaussianBlur((3, 3), p=0.03),
        ]
    else:
        tv_pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=interpolation),
            CenterCropOrPad(image_size, fill=fill),
        ]

    alb_pp += [
        alb.Normalize(image_mean, image_std),
        alb.pytorch.ToTensorV2(),
    ]
    tv_pp += [alb_wrapper(alb.Compose(alb_pp))]
    return transforms.Compose(tv_pp)


def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]

    return f


class CropMargin:

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            assert False
        else:
            data = np.array(img.convert("L"))
            data = data.astype(np.uint8)
            max_val = data.max()
            min_val = data.min()
            if max_val == min_val:
                return img
            data = (data - min_val) / (max_val - min_val) * 255
            gray = 255 * (data < 200).astype(np.uint8)

            coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
            a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
            return img.crop((a, b, w + a, h + b))


class AlignLongAxis:
    def __init__(
            self,
            input_size,
            interpolation=transforms.InterpolationMode.BICUBIC
    ):
        self.input_size = input_size
        self.interpolation = interpolation

    def __call__(self, img):
        is_tensor = isinstance(img, torch.Tensor)
        img_height, img_width = img.shape[-2:] if is_tensor else (img.height, img.width)
        if (
            (self.input_size[0] > self.input_size[1] and img_width > img_height) or
            (self.input_size[0] < self.input_size[1] and img_width < img_height)
        ):
            img = F.rotate(img, angle=-90, expand=True, interpolation=self.interpolation)
        return img


class RandomPad:
    def __init__(self, input_size, fill=0):
        self.input_size = input_size
        self.fill = fill

    @staticmethod
    def get_params(img, input_size):
        width, height = F.get_image_size(img)
        delta_width = max(input_size[1] - width, 0)
        delta_height = max(input_size[0] - height, 0)
        pad_left = random.randint(0, delta_width)
        pad_top = random.randint(0, delta_height)
        pad_right = delta_width - pad_left
        pad_bottom = delta_height - pad_top
        return (
            pad_left,
            pad_top,
            pad_right,
            pad_bottom
        )

    def __call__(self, img):
        padding = self.get_params(img, self.input_size)
        img = F.pad(img, padding, self.fill)
        return img


class ResizeKeepRatio:
    """ Resize and Keep Ratio
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation='bilinear',
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = timm.data.transforms.str_to_interp_mode(interpolation)
        self.longest = float(longest)
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        """Get parameters
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        interpolate_str = timm.data.transforms.interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={interpolate_str})'
        format_string += f', longest={self.longest:.3f})'
        return format_string


class Bitmap:
    def __init__(self, threshold=200):
        self.lut = [0 if i < threshold else i for i in range(256)]

    def __call__(self, img):
        if img.mode == "RGB" and len(self.lut) == 256:
            lut = self.lut + self.lut + self.lut
        else:
            lut = self.lut
        return img.point(lut)


if has_albumentations:

    class ErosionAlb(alb.ImageOnlyTransform):
        def __init__(self, scale, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            if type(scale) is tuple or type(scale) is list:
                assert len(scale) == 2
                self.scale = scale
            else:
                self.scale = (scale, scale)

        def apply(self, img, **params):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
            )
            img = cv2.erode(img, kernel, iterations=1)
            return img


    class DilationAlb(alb.ImageOnlyTransform):
        def __init__(self, scale, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            if type(scale) is tuple or type(scale) is list:
                assert len(scale) == 2
                self.scale = scale
            else:
                self.scale = (scale, scale)

        def apply(self, img, **params):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                tuple(np.random.randint(self.scale[0], self.scale[1], 2))
            )
            img = cv2.dilate(img, kernel, iterations=1)
            return img


    class BitmapAlb(alb.ImageOnlyTransform):
        def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
            super().__init__(always_apply=always_apply, p=p)
            self.lower = lower
            self.value = value

        def apply(self, img, **params):
            img = img.copy()
            img[img < self.lower] = self.value
            return img


