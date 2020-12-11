from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import random

# 来源：https://www.kaggle.com/isaienkov/cassava-leaf-disease-classification-data-analysis/
unusual = ['1004389140.jpg', '1008244905.jpg', '1338159402.jpg', '1339403533.jpg', '159654644.jpg', '1010470173.jpg',
           '1014492188.jpg', '1359893940.jpg', '1366430957.jpg', '1689510013.jpg', '1726694302.jpg', '1770746162.jpg',
           '1773381712.jpg', '1848686439.jpg', '1905119159.jpg', '1917903934.jpg', '1960041118.jpg', '199112616.jpg',
           '2016389925.jpg', '2073193450.jpg', '2074713873.jpg', '2084868828.jpg', '2139839273.jpg', '2166623214.jpg',
           '2262263316.jpg', '2276509518.jpg', '2278166989.jpg', '2321669192.jpg', '2320471703.jpg', '2382642453.jpg',
           '2415837573.jpg', '2482667092.jpg', '2604713994.jpg', '262902341.jpg', '2642216511.jpg', '2698282165.jpg',
           '2719114674.jpg', '274726002.jpg', '2925605732.jpg', '2981404650.jpg', '3040241097.jpg', '3043097813.jpg',
           '3123906243.jpg', '3126296051.jpg', '3199643560.jpg', '3251960666.jpg', '3252232501.jpg', '3425850136.jpg',
           '3435954655.jpg', '3477169212.jpg', '3609350672.jpg', '3652033201.jpg', '3810809174.jpg', '3838556102.jpg',
           '3881028757.jpg', '3892366593.jpg', '4060987360.jpg', '4089218356.jpg', '4134583704.jpg', '4203623611.jpg',
           '421035788.jpg', '4239074071.jpg', '4269208386.jpg', '457405364.jpg', '549854027.jpg', '554488826.jpg',
           '580111608.jpg', '600736721.jpg', '616718743.jpg', '695438825.jpg', '723564013.jpg', '746746526.jpg',
           '826231979.jpg', '847847826.jpg', '9224019.jpg', '992748624.jpg']

def show_unisual(images_number=16):
    plot_list = random.sample(unusual, images_number)
    labels = ['Unusual Cassava Brown Streak Disease' for i in range(len(plot_list))]
    size = np.sqrt(images_number)
    if int(size) * int(size) < images_number:
        size = int(size) + 1

    plt.figure(figsize=(20, 20))

    for ind, (image_id, label) in enumerate(zip(plot_list, labels)):
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(os.path.join('F:/dataset/cassava-leaf-disease-classification/', "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")

    plt.show()

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
    # filter_noise()
    show_unisual()
    pass
