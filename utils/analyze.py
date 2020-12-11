import numpy as np
import pandas as pd
import os
import seaborn as sn
import matplotlib.pyplot as plt
import cv2


'''
0 - CBB - Cassava Bacterial Blight
1 - CBSD - Cassava Brown Streak Disease
2 - CGM - Cassava Green Mottle
3 - CMD - Cassava Mosaic Disease
4 - Healthy
'''


def visualize_batch(image_ids, labels):
    plt.figure(figsize=(16, 12))

    for ind, (image_id, label) in enumerate(zip(image_ids, labels)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Class: {label}", fontsize=12)
        plt.axis("off")

    plt.show()

def show_class():

    # print(df_train.head())

    plt.figure(figsize=(8, 4))
    sn.countplot(y="label", data=df_train)
    plt.show()


if __name__ == '__main__':
    BASE_DIR = 'F:/dataset/cassava-leaf-disease-classification/'
    df_train = pd.read_csv('../datasets/train.csv')
    # show_class()

    # tmp_df = df_train[df_train["label"] == 4]
    # print(f"Total train images for class 0: {tmp_df.shape[0]}")
    #
    # tmp_df = tmp_df.sample(9)
    # image_ids = tmp_df["image_id"].values
    # labels = tmp_df["label"].values
    #
    # visualize_batch(image_ids, labels)

    predict_results = np.load('../output/results/efn_b5_ns_train.npy')
    print(predict_results[:100])
    print(predict_results.shape)
