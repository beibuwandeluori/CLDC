import warnings
warnings.filterwarnings("ignore")
import warnings
from PIL.Image import DecompressionBombWarning
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DecompressionBombWarning)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import *
import time
from networks.models import get_efficientnet, model_selection
from datasets.dataset import CLDCDatasetSubmission, get_valid_transforms
import pandas as pd
from tqdm import tqdm
from utils.utils import Test_time_agumentation
from torch.autograd import Variable


# 9 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img)) if activation is not None else model_(img)
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img)) if activation is not None else model_(flip_img)
    # 2*3=6
    for flip_img in [img, flip_imgs[0]]:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += activation(model_(rot_flip_img)) if activation is not None else model_(rot_flip_img)
    outputs /= 9.0

    return outputs

def test_model():
    model.eval()
    num_steps = len(eval_loader)
    print(f'total batches: {num_steps}')
    preds = []
    image_names = []
    with torch.no_grad():
        for i, (XI, image_name) in enumerate(tqdm(eval_loader)):
            # if i % 50 == 0:
            #     print(i, i/len(eval_loader))

            x = Variable(XI.cuda(device_id))
            if is_tta:
                output = TTA(model_=model, img=x, activation=nn.Softmax(dim=1))
            else:
                output = model(x)
                output = nn.Softmax(dim=1)(output)
            confs, predicts = torch.max(output.detach(), dim=1)
            preds += list(predicts.cpu().numpy())
            image_names += list(image_name)

    print(len(preds), len(image_names))
    # file = pd.read_csv('/data1/cby/py_project/CLDC/output/result/pred_results.csv')
    # test_df = pd.DataFrame(file)
    # for pred, name in zip(preds, image_names):
    #     test_df.iloc[list(test_df['image_name']).index(name), 1] = pred
    # test_df.to_csv('./output/result/submission_efn-b0.csv', index=False)

    with open(csv_path, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds))):
            f.write('{0},{1}\n'.format(image_names[i], preds[i]))


if __name__ == '__main__':
    is_tta = True
    csv_path = './output/result/submission_efn-b1_512_e45.csv'
    test_batch_size = 128
    is_alb = False
    device_id = 0
    model_name = 'efficientnet-b1'  # 'resnet50'
    # model_path = None
    model_path = '/data1/cby/py_project/CLDC/output/weights/efficientnet-b1_LS_512/efn-b0_45_acc_0.5447.pth'
    model = get_efficientnet(model_name=model_name)
    # model, *_ = model_selection(modelname=model_name, num_out_classes=5000, dropout=None)
    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)

    start = time.time()
    epoch_start = 1
    num_epochs = 1
    xdl_test = CLDCDatasetSubmission(is_one_hot=False, transforms=get_valid_transforms(size=512, is_alb=is_alb),
                                     is_alb=is_alb)
    eval_loader = DataLoader(xdl_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_dataset_len = len(xdl_test)
    print('test_dataset_len:', test_dataset_len)
    test_model()
    print('Total time:', time.time() - start)







