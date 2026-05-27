import random

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
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def __call__(self, image, target):
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        image = F.adjust_brightness(image, factor)
        return image, target


def get_transforms(cfg, split="train"):
    transforms = []

    if split == "train":
        aug_cfg = cfg.get("augmentation", {}).get("train", {})

        flip_cfg = aug_cfg.get("horizontal_flip", {})
        if flip_cfg.get("enabled", False):
            transforms.append(RandomHorizontalFlip(prob=flip_cfg.get("prob", 0.5)))

        bright_cfg = aug_cfg.get("random_brightness", {})
        if bright_cfg.get("enabled", False):
            transforms.append(RandomBrightnessContrast(brightness=bright_cfg.get("value", 0.2)))

    transforms.append(ToTensor())
    return Compose(transforms)
