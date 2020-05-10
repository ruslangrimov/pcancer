import albumentations as alb

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


def augment_v1_clr_only(imgs):
    return _augment_clr_only(imgs, _aug_v1_clr_only)
