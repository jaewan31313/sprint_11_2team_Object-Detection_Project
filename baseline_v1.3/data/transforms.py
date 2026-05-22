import random

import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width = image.width if hasattr(image, "width") else image.shape[-1]
            image = F.hflip(image)
            if target is not None and "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, target):
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = F.adjust_brightness(image, factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = F.adjust_contrast(image, factor)
        return image, target


class Normalize:
    """ImageNet normalization — Faster R-CNN handles this internally, but
    useful when swapping backbones."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transforms(cfg, split="train"):
    aug_cfg = cfg["augmentation"].get(split, {})
    transforms = []

    if split == "train":
        flip_prob = aug_cfg.get("horizontal_flip", 0.5)
        if flip_prob > 0:
            transforms.append(RandomHorizontalFlip(prob=flip_prob))

        brightness = aug_cfg.get("random_brightness", 0.0)
        if brightness > 0:
            transforms.append(RandomBrightnessContrast(brightness=brightness))

    transforms.append(ToTensor())
    return Compose(transforms)
