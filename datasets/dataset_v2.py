import sys
sys.path.append('/data1/cby/py_project/CLDC/datasets/FMix-master')
from glob import glob
import torch
import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from fmix import sample_mask, make_low_freq_image, binarise_mask


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2


def get_train_transforms(size=512):
    return Compose([
        RandomResizedCrop(size, size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms(size=512):
    return Compose([
        CenterCrop(size, size, p=1.),
        Resize(size, size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, df, data_root='/raid/chenby/cassava-leaf-disease-classification/train_images',
                 transforms=None, output_label=True, one_hot_label=False, do_fmix=False,
                 do_cutmix=False, input_size=512):
        super().__init__()
        fmix_params = {
                          'alpha': 1.,
                          'decay_power': 3.,
                          'shape': (input_size, input_size),
                          'max_soft': True,
                          'reformulate': False
                      },
        cutmix_params = {
            'alpha': 1,
        }
        self.input_size = input_size
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label:
            self.labels = self.df['label'].values
            # print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max() + 1)[self.labels]
                # print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
        img_path = os.path.join(self.data_root, self.df.loc[index]['image_id'])
        img = get_img(img_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.random() > 0.5:
            with torch.no_grad():
                # lam, mask = sample_mask(**self.fmix_params)

                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']), 0.6, 0.7)

                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])

                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                # mix image
                img = mask_torch * img + (1. - mask_torch) * fmix_img

                # print(mask.shape)

                # assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum() / self.input_size / self.input_size
                target = rate * target + (1. - rate) * self.labels[fmix_ix]
                # print(target, mask, img)
                # assert False

        if self.do_cutmix and np.random.random() > 0.5:
            # print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((self.input_size, self.input_size), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]
                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.input_size * self.input_size))
                print(target, self.labels[cmix_ix], type(target), type(self.labels[cmix_ix]), type(rate), type(1. - rate))
                target = rate * target + (1. - rate) * self.labels[cmix_ix]

            # print('-', img.sum())
            # print(target)
            # assert False

        # do label smoothing
        # print(type(img), type(target))
        if self.output_label:
            return img, target
        else:
            return img


if __name__ == '__main__':
    # load_image_paths_labels(k=0)

    start = time.time()
    train = pd.read_csv('/raid/chenby/cassava-leaf-disease-classification/train.csv')
    xdl = CassavaDataset(df=train,  transforms=get_train_transforms(size=512), output_label=True, one_hot_label=True, do_fmix=False,
                         do_cutmix=True, input_size=512)

    print('length:', len(xdl))
    train_loader = DataLoader(xdl, batch_size=32, shuffle=False, num_workers=4)
    for i, (img, label) in enumerate(train_loader):
        print(i, '/', len(train_loader), img.shape, label.shape, label[0])
        if i == 20:
            break
    end = time.time()
    print('end iterate')
    print('DataLoader total time: %fs' % (end - start))

    pass
