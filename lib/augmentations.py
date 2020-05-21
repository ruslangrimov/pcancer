import random

import numpy as np
import albumentations as alb
import staintools

_aug_v1 = alb.Compose([
    alb.RandomRotate90(),
    alb.Flip(),
    alb.Transpose(),
    alb.ShiftScaleRotate(shift_limit=0.0625,
                         scale_limit=0.2,
                         rotate_limit=0,
                         p=0.5),
    # alb.CLAHE(clip_limit=2),
    alb.RandomContrast(p=0.2),
    alb.RandomBrightness(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.HueSaturationValue(p=0.2),
    alb.RGBShift(p=0.2),
    alb.OneOf([
        alb.IAAAdditiveGaussianNoise(),
        alb.GaussNoise(),
    ], p=0.02),
    alb.OneOf([
        alb.MotionBlur(p=0.2),
        alb.MedianBlur(blur_limit=11, p=0.2),
        alb.Blur(blur_limit=11, p=0.2),
    ], p=0.5),
], p=0.9)

_aug_v1_wo_blur = alb.Compose([
    alb.RandomRotate90(),
    alb.Flip(),
    alb.Transpose(),
    alb.ShiftScaleRotate(shift_limit=0.0625,
                         scale_limit=0.2,
                         rotate_limit=0,
                         p=0.5),
    # alb.CLAHE(clip_limit=2),
    alb.RandomContrast(p=0.3),
    alb.RandomBrightness(p=0.3),
    alb.RandomGamma(p=0.3),
    alb.HueSaturationValue(p=0.3),
    alb.RGBShift(p=0.3)
], p=0.9)

_aug_v1_wo_blur_and_clr = alb.Compose([
    alb.RandomRotate90(),
    alb.Flip(),
    alb.Transpose(),
    alb.ShiftScaleRotate(shift_limit=0.0625,
                         scale_limit=0.2,
                         rotate_limit=0,
                         p=0.5)
], p=0.9)

_aug_v1_only_scale = alb.Compose([
    alb.ShiftScaleRotate(shift_limit=0.0,
                         scale_limit=0.2,
                         rotate_limit=0,
                         p=0.5)
], p=0.9)

_aug_v1_clr_only = alb.Compose([
    alb.ShiftScaleRotate(shift_limit=0.0,  # ToDo: Not sure
                         scale_limit=0.2,
                         rotate_limit=0,
                         p=0.5),
    alb.RandomContrast(p=0.2),
    alb.RandomBrightness(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.HueSaturationValue(p=0.2),
    alb.RGBShift(p=0.2),
    alb.OneOf([
        alb.IAAAdditiveGaussianNoise(),
        alb.GaussNoise(),
    ], p=0.02),
    alb.OneOf([
        alb.MotionBlur(p=0.2),
        alb.MedianBlur(blur_limit=11, p=0.2),
        alb.Blur(blur_limit=11, p=0.2),
    ], p=0.5),
], p=0.9, additional_targets={f'image{i}': 'image' for i in range(1000)})


normalizer = staintools.StainNormalizer(method='macenko')


def normalize_he(img, mask):
    img = np.clip(normalizer.transform(img), 0, 255)

    img = img / 255

    return img, mask


def augment_he(imgs, mask):
    # Concatenate imgs in one line and process it together to apply the same augmentations to all
    raise NotImplementedError("Not implemented for wsi yet")


def augment_he_D8(img, mask):
    augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.4,
                                          sigma2=0.4, augment_background=False)
    augmentor.fit(img)

    img = np.clip(augmentor.pop(), 0, 255).astype(np.uint8)

    return _augment(img, mask, _aug_v1_wo_blur_and_clr)


def augment_empty(img, mask):
    img = img / 255

    return img, mask


def _augment(img, mask, aug):
    augmented = aug(image=img, mask=mask)

    img = augmented['image']
    mask = augmented['mask']

    img = img / 255

    return img, mask


def augment_empty_clr_only(imgs):
    return [img / 255 for img in imgs]


def _augment_clr_only(imgs, aug):
    imgs = {f'image{i}': imgs[i] for i in range(len(imgs))}
    imgs['image'] = imgs['image0']
    del imgs['image0']

    imgs = aug(**imgs)
    imgs['image0'] = imgs['image']
    del imgs['image']

    imgs = [imgs[f'image{i}'] / 255 for i in range(len(imgs))]

    return imgs


def augment_v1(img, mask):
    return _augment(img, mask, _aug_v1)


def augment_v1_he_clr_mix(img, mask):
    return augment_he_D8 if random.random() > 0.5 else augment_v1(img, mask)


def augment_v1_wo_blur(img, mask):
    return _augment(img, mask, _aug_v1_wo_blur)


def augment_v1_he_clr_wo_blur_mix(img, mask):
    return (augment_he_D8 if random.random() > 0.5 else
            augment_v1_wo_blur(img, mask))


def augment_v1_clr_only(imgs):
    return _augment_clr_only(imgs, _aug_v1_clr_only)
