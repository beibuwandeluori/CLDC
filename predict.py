from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from datasets.dataset import get_valid_transforms, CLDCDatasetSubmission
from networks.models import CassvaImgClassifier
from utils.utils import Test_time_agumentation
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import pandas as pd
import time


# 3 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img))
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img))
    # 2*3=6
    #     for flip_img in [img, flip_imgs[0]]:
    #         rot_flip_imgs = tta.tensor_rotation(flip_img)
    #         for rot_flip_img in rot_flip_imgs:
    #             outputs += activation(model_(rot_flip_img))

    outputs /= 3

    return outputs


def predict(model, test_dataset, batch_size=64, num_workers=4, is_TTA=False):
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False,
                             drop_last=False)
    tk0 = tqdm(test_loader)
    model.eval()
    preds = []
    with torch.no_grad():
        for i, (im, img_name) in enumerate(tk0):
            im = Variable(im.cuda(device_id))
            if is_TTA:
                outputs = TTA(model, im)
                preds.extend(F.softmax(outputs, 1).cpu().numpy())
            else:
                outputs = model(im)
                outputs = F.softmax(outputs, 1).cpu().numpy()
                preds.extend(outputs)
    preds = np.array(preds)

    return preds


if __name__ == '__main__':
    models = [
        CassvaImgClassifier(model_arch='tf_efficientnet_b5_ns', n_class=5),
        # CassvaImgClassifier(model_arch='tf_efficientnet_b5_ns', n_class=5),
        # CassvaImgClassifier(model_arch='tf_efficientnet_b5_ns', n_class=5),
        # CassvaImgClassifier(model_arch='tf_efficientnet_b5_ns', n_class=5),
        # CassvaImgClassifier(model_arch='tf_efficientnet_b5_ns', n_class=5)
    ]
    model_paths = ['/data1/cby/py_project/CLDC/output/weights/tf_efficientnet_b5_ns_0_512/e25_acc_0.8895.pth',
                   # '/data1/cby/py_project/CLDC/output/weights/tf_efficientnet_b5_ns_1_512/e5_acc_0.8913.pth',
                   # '/data1/cby/py_project/CLDC/output/weights/tf_efficientnet_b5_ns_2_512/e7_acc_0.8976.pth',
                   # '/data1/cby/py_project/CLDC/output/weights/tf_efficientnet_b5_ns_3_512/e7_acc_0.8867.pth',
                   # '/data1/cby/py_project/CLDC/output/weights/tf_efficientnet_b5_ns_4_512/e6_acc_0.8903.pth'
                   ]  # 0.9

    batch_size = 64
    num_workers = 4
    device_id = 2
    is_TTA = False
    is_alb = False
    input_size = 512
    # root_path = '/raid/chenby/cassava-leaf-disease-classification/test_images'
    root_path = '/raid/chenby/cassava-leaf-disease-classification/train_images'
    csv_path = './output/results/efn_b5_ns_train.csv'
    npy_path = './output/results/efn_b5_ns_train.npy'

    image_names = sorted(os.listdir(root_path))
    # print(np.array(image_names)[:100])
    train = pd.read_csv('/data1/cby/py_project/CLDC/datasets/train.csv')
    # print(train.image_id.values[:100])
    # for i in range(100):
    #     print(np.array(image_names)[i], train.image_id.values[i], np.array(image_names)[i] == train.image_id.values[i])

    test_dataset = CLDCDatasetSubmission(root_path=root_path, is_alb=is_alb,
                                         transforms=get_valid_transforms(size=input_size, is_alb=is_alb))
    start = time.time()

    for i in range(len(models)):
        model = models[i]
        model.load_state_dict(torch.load(model_paths[i], map_location='cpu'))
        print('Model{} found in {}'.format(i, model_paths[i]))
        model = model.cuda(device_id)
        if i == 0:
            preds = predict(model=model, test_dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                            is_TTA=is_TTA)
        else:
            preds += predict(model=model, test_dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                             is_TTA=is_TTA)

        del model
        torch.cuda.empty_cache()

    preds = preds / len(models)
    pred_labels = preds.argmax(axis=1)
    # print(pred_labels)
    print('validation accuracy = {:.5f}'.format((train.label.values == pred_labels).mean()))
    # np.save(npy_path, preds)
    print(preds[:100])
    image_names = sorted(os.listdir(root_path))
    # with open(csv_path, 'w') as f:
    #     f.write('{0},{1}\n'.format('image_id', 'label'))
    #     for i in tqdm(range(len(pred_labels))):
    #         f.write('{0},{1}\n'.format(image_names[i], pred_labels[i]))

    print('total time:{:.4f}'.format(time.time() - start))
