from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import numpy as np
import os
from tqdm import tqdm

def read_image(image_path):
    try:
        image = Image.open(image_path)
        print(np.array(image).shape, image.mode)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = np.array(image, dtype=np.float32)
    except:
        try:  # 可能有些图片在Image读取出错，可以cv2读
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        except:
            print(image_path)
    # print(image.shape)
    image = cv2.resize(image, (10, 10))
    image = np.reshape(image, (10 * 10 * 3,))
    return image

def clean_images(image_paths):
    print(image_paths)
    image_names = sorted(os.listdir(image_paths))
    X = []
    for i in image_names:
        img = read_image(os.path.join(image_paths, i))
        # print(gray.shape)
        X.append(img)
    X = np.array(X)
    print(X.shape)

    pca = PCA(n_components=16)
    newX = pca.fit_transform(X)  # 等价于pca.fit(X) pca.transform(X)
    # print(X)
    # print(newX)
    # print(invX)
    print(sum(pca.explained_variance_ratio_))
    n_clusters = 10
    y_preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(newX)

    # for i in range(len(image_names)):
    #     print(i, image_names[i], y_preds[i])
    print(y_preds, [np.sum(y_preds == i) for i in range(n_clusters)])

def data_analyse(root_path='/data1/cby/dataset/CLDC/train'):
    class_names = sorted(os.listdir(root_path))
    print(len(class_names))
    class_lengths = []
    for class_name in class_names:
        class_path = os.path.join(root_path, class_name)
        image_names = os.listdir(class_path)
        print(class_name, len(image_names))
        class_lengths.append(len(image_names))


def resize(image, size=512):
    is_resize = False
    width, height = image.size
    if width > size or height > size:
        is_resize = True
        if width > height:
            width_new = size
            height_new = int(height * (size/width))
        else:
            width_new = int(width * (size/height))
            height_new = size
        image = image.resize((width_new, height_new), Image.ANTIALIAS)

    return image, is_resize


def resize_imgs(is_train=True):
    if not is_train:
        test_root = '/raid/chenby/CLDC/test'
        image_paths = [os.path.join(test_root, name) for name in os.listdir(test_root)]
    else:
        image_paths = np.load('/data1/cby/py_project/CLDC/datasets/npy/train_paths.npy')

    for i, image_path in enumerate(tqdm(sorted(image_paths)[:437149-300000])):
        if is_train:
            save_path = image_path.replace('train', 'train_512')
        else:
            save_path = image_path.replace('test', 'test_512')
        if os.path.exists(save_path):
            continue
        image = Image.open(image_path)
        image_new, is_resize = resize(image, size=512)
        # print(image.size, np.array(image).shape, image_new.size, is_resize)
        path, name = os.path.split(save_path)
        # print(save_path, path, name)
        if not os.path.exists(path):
            os.makedirs(path)
        image_new.save(save_path)
        # if i == 10:
        #     break


if __name__ == '__main__':
    # clean_images(image_paths='/data1/cby/dataset/CLDC/train/2290')
    # data_analyse()
    resize_imgs(is_train=False)
    pass
