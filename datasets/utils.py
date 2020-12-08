from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import hashlib


def filter_noise():
    root_path = '/raid/chenby/cassava-leaf-disease-classification/train_images'
    csv_path = '/raid/chenby/cassava-leaf-disease-classification/train.csv'
    csv_data = pd.read_csv(csv_path)
    image_paths = []
    image_labels = []
    for index, row in csv_data.iterrows():
        image_paths.append(os.path.join(root_path, row['image_id']))
        image_labels.append(row['label'])

    images_md5 = []
    for i, path in enumerate(image_paths):
        fd = np.array(Image.open(path))
        fmd5 = hashlib.md5(fd)
        if i % 1000 == 0:
            print(i, os.path.basename(path), fmd5.hexdigest())
        images_md5.append(fmd5.hexdigest())

    print(len(image_paths), len(images_md5), len(set(images_md5)))


if __name__ == '__main__':
    # clean_images(image_paths='/data1/cby/dataset/CLDC/train/2290')
    filter_noise()
    pass
