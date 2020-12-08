import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from PIL import Image
import time
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Normalize, RandomBrightnessContrast, RandomResizedCrop,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize, ImageCompression, Rotate, Transpose,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, Cutout, GridDropout,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip, GaussianBlur, CoarseDropout,
    PadIfNeeded, ToGray, FancyPCA)
from catalyst.data.sampler import BalanceClassSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold


def get_train_transforms(size=300, is_alb=True):
    if is_alb:
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
        ])
    else:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), fillcolor=0, scale=(0.8, 1.2), shear=None),
            transforms.Resize((int(size / 0.875), int(size / 0.875))),
            transforms.RandomCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])


def get_valid_transforms(size=300, is_alb=True):
    if is_alb:
        return Compose([
                CenterCrop(size, size, p=1.),
                Resize(size, size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)
    else:
        return transforms.Compose([
            transforms.Resize((int(size / 0.875), int(size / 0.875))),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def one_hot(size, target):
    vec = np.zeros(size, dtype=np.float32)
    vec[target] = 1.
    return vec


def load_image_paths_labels(root_path='/raid/chenby/cassava-leaf-disease-classification/train_images',
                            csv_path='/raid/chenby/cassava-leaf-disease-classification/train.csv',
                            data_type='train', k=-1, n_splits=5):
    csv_data = pd.read_csv(csv_path)
    image_paths = []
    image_labels = []
    for index, row in csv_data.iterrows():
        image_paths.append(os.path.join(root_path, row['image_id']))
        image_labels.append(row['label'])

    if k != -1 and k < n_splits:
        kf = KFold(n_splits=n_splits)
        for i, (train, valid) in enumerate(kf.split(X=image_paths, y=image_labels)):
            # print(i, "train:%d,valid:%d" % (len(train), len(valid)))
            # print(train[:5], valid[:5])
            if i == k:
                train_indexs, valid_indexs = train, valid
        x_train = np.array(image_paths)[train_indexs]
        x_test = np.array(image_paths)[valid_indexs]
        y_train = np.array(image_labels)[train_indexs]
        y_test = np.array(image_labels)[valid_indexs]
        # print(np.unique(y_train), np.unique(y_test))
    else:
        x_train, x_test, y_train, y_test = train_test_split(image_paths, image_labels, test_size=0.2, random_state=2020)
    print('train:', len(x_train), 'test:', len(x_test), 'total:', len(x_test) + len(x_train))
    if data_type == 'train':
        return x_train, y_train
    else:
        return x_test, y_test


def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


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


class CLDCDataset(Dataset):

    def __init__(self, root_path='/raid/chenby/cassava-leaf-disease-classification/train_images',
                 csv_path='/raid/chenby/cassava-leaf-disease-classification/train.csv',
                 is_one_hot=False, transform=None, do_cutmix=False,
                 classes_num=5, data_type='train', is_alb=True, k=-1):
        super().__init__()
        self.classes_num = classes_num
        self.transform = transform
        self.is_one_hot = is_one_hot
        self.is_alb = is_alb
        self.do_cutmix = do_cutmix
        self.data_type = data_type
        self.images, self.labels = load_image_paths_labels(root_path=root_path, csv_path=csv_path,
                                                           data_type=data_type, k=k)
        print(data_type, len(self.images), self.labels.shape)
        # if is_one_hot:
        #     self.labels = np.eye(self.classes_num)[self.labels]

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_path = self.images[index]
        image = read_image(image_path)
        if self.transform and self.is_alb:
            image = image.astype(np.float32)
            image = self.transform(image=image)['image']
        elif self.transform and not self.is_alb:
            image = self.transform(Image.fromarray(image))

        if self.is_one_hot:
            label = one_hot(self.classes_num, label)

        if self.data_type == 'train' and self.do_cutmix and np.random.random() > 0.5:
            image, label = self.cut_mix(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def get_labels(self):
        return list(self.labels)

    def cut_mix(self, img, label):
        cmix_ix = np.random.choice(len(self.images), size=1)[0]
        cmix_image_path = self.images[cmix_ix]
        cmix_img = read_image(cmix_image_path)

        if self.transform and self.is_alb:
            cmix_img = cmix_img.astype(np.float32)
            cmix_img = self.transform(image=cmix_img)['image']
        elif self.transform and not self.is_alb:
            cmix_img = self.transform(Image.fromarray(cmix_img))

        lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
        bbx1, bby1, bbx2, bby2 = rand_bbox((img.shape[0], img.shape[1]), lam)

        img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

        rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[0] * img.shape[1]))

        cmix_label = self.labels[cmix_ix]
        if self.is_one_hot:
            cmix_label = one_hot(self.classes_num, cmix_label)
        # print(label, cmix_label, type(label), type(cmix_label), type(rate))
        target = rate * label + (1. - rate) * cmix_label

        return img, target


class CLDCDatasetSubmission(Dataset):

    def __init__(self, root_path='/raid/chenby/cassava-leaf-disease-classification/test_images', is_one_hot=False, transforms=None,
                 classes_num=5, is_alb=True):
        super().__init__()
        self.classes_num = classes_num
        self.transforms = transforms
        self.is_one_hot = is_one_hot
        self.is_alb = is_alb
        self.image_names = sorted(os.listdir(root_path))
        self.images = []
        for name in self.image_names:
            self.images.append(os.path.join(root_path, name))

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = read_image(image_path)
        image_name = self.image_names[index]
        # print(image_name)
        if self.transforms and self.is_alb:
            image = image.astype(np.float32)
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        elif self.transforms and not self.is_alb:
            image = self.transforms(Image.fromarray(image))

        return image, image_name

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    # load_image_paths_labels(k=0)

    start = time.time()
    xdl = CLDCDataset(transform=get_valid_transforms(size=224, is_alb=False), is_one_hot=True, data_type='train',
                      is_alb=False, k=0, do_cutmix=True)

    print('length:', len(xdl))
    train_loader = DataLoader(xdl, batch_size=32, shuffle=False, num_workers=4)
    # train_loader = DataLoader(xdl, batch_size=128, shuffle=False, num_workers=4,
    #                           sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="downsampling"))
    for i, (img, label) in enumerate(train_loader):
        print(i, '/', len(train_loader), img.shape, label.shape, label[0])
        if i == 20:
            break
    end = time.time()
    print('end iterate')
    print('DataLoader total time: %fs' % (end - start))

    pass

