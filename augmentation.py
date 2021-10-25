import random
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def augGeo():
    return iaa.SomeOf((0,3),[
        iaa.CropAndPad(percent=(-0.10, 0.10), pad_cval=-1024),
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, mode="constant", cval=-1024, order=[0,1]),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode="constant", cval=-1024, order=[0,1]),
        iaa.Affine(rotate=(-45, 45), mode="constant", cval=-1024, order = [0,1]),
        iaa.Affine(shear=(-15, 15))
    ],random_order=True)

def augmenters():
    return iaa.SomeOf((0,3),[
        iaa.GaussianBlur(sigma=(0.0, 0.3)),
        iaa.MedianBlur(k=(3, 5)),
        iaa.AdditiveGaussianNoise(scale=(0, .05*255)),
        iaa.AdditiveLaplaceNoise(scale=(0, .05*255)),
        iaa.Dropout(p=(0, .1)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sharpen(alpha=(0.0, 0.75), lightness=(0.75, 2.0)),
    ], random_order=True)

def batchAugmented(imgBatch, augment=True):
    ia.seed(random.randint(0,100))
    imgBatch = np.asarray(imgBatch)
    if augment:
        aug = augmenters().to_deterministic()
        aug_geo = augGeo().to_deterministic()
        imgBatch = imgBatch.astype(dtype=np.uint8)
        imgBatch = aug.augment_image(imgBatch)
        imgBatch = aug_geo.augment_image(imgBatch)

    return imgBatch