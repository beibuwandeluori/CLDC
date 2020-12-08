import torch
import pretrainedmodels
import torch.nn as nn
from torch.nn import init
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
import timm

# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel//3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)

class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0, is_DCT=False):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        self.is_DCT = is_DCT
        self.input_channel = 3
        if self.is_DCT:
            self.DCT = DCT_Layer()
            self.input_channel = 48
        if modelchoice == 'resnet50' or modelchoice == 'resnet18' or modelchoice == 'resnet101' or modelchoice == 'resnet152'\
                or modelchoice == 'resnext101_32x8d' or modelchoice == 'resnext50_32x4d':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            if modelchoice == 'resnet101':
                self.model = torchvision.models.resnet101(pretrained=True)
            if modelchoice == 'resnext101_32x8d':
                self.model = torchvision.models.resnext101_32x8d(pretrained=True)
            if modelchoice == 'resnext50_32x4d':
                self.model = torchvision.models.resnext50_32x4d(pretrained=True)
            # replace first Conv2d
            if self.input_channel != 3:
                conv1_weight = self.model.conv1.weight
                self.model.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.conv1.weight = init_imagenet_weight(conv1_weight, self.input_channel)

            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
                init.normal_(self.model.fc.weight.data, std=0.001)
                init.constant_(self.model.fc.bias.data, 0.0)
            else:
                self.model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                init.normal_(self.model.fc[2].weight.data, std=0.001)
                init.constant_(self.model.fc[2].bias.data, 0.0)
        elif modelchoice == 'se_resnext101_32x4d' or modelchoice == 'se_resnext50_32x4d':
            if modelchoice == 'se_resnext101_32x4d':
                self.model = pretrainedmodels.se_resnext101_32x4d(pretrained='imagenet')

            if modelchoice == 'se_resnext50_32x4d':
                self.model = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet')

            # replace first Conv2d
            self.model.layer0.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
                # init.normal_(self.model.last_linear.weight.data, std=0.001)
                # init.constant_(self.model.last_linear.bias.data, 0.0)
            else:
                print('Using dropout', dropout, num_ftrs)
                # self.model.last_linear = nn.Sequential(
                #     nn.Dropout(p=dropout),
                #     nn.Linear(num_ftrs, num_out_classes)
                # )
                self.model.last_linear = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.BatchNorm1d(256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                # weights init
                init.kaiming_normal_(self.model.last_linear[0].weight.data, a=0, mode='fan_out')
                init.constant_(self.model.last_linear[0].bias.data, 0.0)
                init.normal_(self.model.last_linear[1].weight.data, 1.0, 0.02)
                init.constant_(self.model.last_linear[1].bias.data, 0.0)
                init.normal_(self.model.last_linear[3].weight.data, std=0.001)
                init.constant_(self.model.last_linear[3].bias.data, 0.0)
        elif modelchoice == 'efficientnet-b7' or modelchoice == 'efficientnet-b6'\
                or modelchoice == 'efficientnet-b5' or modelchoice == 'efficientnet-b4'\
                or modelchoice == 'efficientnet-b3' or modelchoice == 'efficientnet-b2'\
                or modelchoice == 'efficientnet-b1' or modelchoice == 'efficientnet-b0':
            # self.model = EfficientNet.from_name(modelchoice, override_params={'num_classes': num_out_classes})
            self.model = get_efficientnet(model_name=modelchoice, num_classes=num_out_classes)
            if self.input_channel != 3:
                # print(self.input_channel)
                self.model._conv_stem.in_channels = self.input_channel
                self.model._conv_stem.weight = init_imagenet_weight(self.model._conv_stem.weight, self.input_channel)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def forward(self, x):
        if self.is_DCT:
            x = self.DCT(x)
        x = self.model(x)
        return x

def model_selection(modelname, num_out_classes, dropout=None, is_DCT=False):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT), 299, \
               True, ['image'], None
    # torchvision
    elif modelname == 'resnet18' or modelname == 'resnet50' or modelname == 'resnet101' or modelname == 'resnet152'\
            or modelname == 'resnext101_32x8d' or modelname == 'resnext50_32x4d':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT), \
               224, True, ['image'], None
    # pretrainedmodels
    elif modelname == 'se_resnext101_32x4d' or modelname == 'se_resnext50_32x4d':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT), \
               224, True, ['image'], None
    elif modelname == 'efficientnet-b7' or modelname == 'efficientnet-b6'\
            or modelname == 'efficientnet-b5' or modelname == 'efficientnet-b4' \
            or modelname == 'efficientnet-b3' or modelname == 'efficientnet-b2' \
            or modelname == 'efficientnet-b1' or modelname == 'efficientnet-b0':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes, is_DCT=is_DCT), \
               224, True, ['image'], None

    else:
        raise NotImplementedError(modelname)

# 自定义DCT Layer, 可以即插即用
class DCT_Layer(nn.Module):
    def __init__(self,):
        super(DCT_Layer, self).__init__()
        self.dct = nn.Conv2d(1, 16, kernel_size=4, padding=2, bias=False)
        self.dct.weight = self.init_DCT()

    # DCT Keras
    def init_DCT(self, shape=(4, 4, 1, 16)):
        PI = math.pi
        DCT_kernel = np.zeros(shape, dtype=np.float32)  # [height,width,input,output], shape=(4, 4, 1, 16)
        u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
        u[0] = math.sqrt(1.0 / 4.0)
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 4):
                    for l in range(0, 4):
                        DCT_kernel[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
                            PI / 8.0 * l * (2 * j + 1))
        DCT_kernel = DCT_kernel.transpose(3, 2, 0, 1)
        dct_weight = nn.Parameter(torch.Tensor(DCT_kernel).view(16, 1, 4, 4), requires_grad=False)

        return dct_weight

    # Trancation operation for DCT
    @staticmethod
    def DCT_Trunc(x):
        trunc = -(F.relu(-x + 8) - 8)
        return trunc

    def forward(self, x):
        for i in range(x.size(1)):
            out = self.dct(x[:, i:i+1, :, :])
            out = self.DCT_Trunc(torch.abs(out))
            if i == 0:
                outs = out
            else:
                outs = torch.cat([outs, out], dim=1)

        # 恢复原来的尺寸
        diffY = x.size()[2] - outs.size()[2]
        diffX = x.size()[3] - outs.size()[3]

        outs = F.pad(outs, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        # print(outs.shape)
        return outs


def get_efficientnet(model_name='efficientnet-b0', num_classes=5, pretrained=True):
    if pretrained:
        net = EfficientNet.from_pretrained(model_name)
    else:
        net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


class DCT_EfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', num_classes=5, base_model_path=None):
        super(DCT_EfficientNet, self).__init__()
        self.DCT = DCT_Layer()
        self.input_channel = 48
        self.efficient = get_efficientnet(model_name=model_name, num_classes=num_classes)
        if base_model_path is not None:
            self.efficient.load_state_dict(torch.load(base_model_path, map_location='cpu'))
            print('Model found in {}'.format(base_model_path))
        else:
            print('No base model found, initializing random model.')

        self.efficient._conv_stem.in_channels = self.input_channel
        self.efficient._conv_stem.weight = init_imagenet_weight(self.efficient._conv_stem.weight, self.input_channel)

    def forward(self, x):
        x = self.DCT(x)
        x = self.efficient(x)
        return x


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch='tf_efficientnet_b3_ns', n_class=5, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # model, image_size = get_efficientnet(model_name='efficientnet-b0', num_classes=5000, pretrained=True), 224
    # model, image_size, *_ = model_selection(modelname='resnet50', num_out_classes=5000, dropout=None)
    # model, image_size = ConcatModel(model_name='efficientnet-b0'), 224
    # model, image_size = DCT_EfficientNet(model_name='efficientnet-b0', num_classes=5000), 224
    model, image_size = CassvaImgClassifier(model_arch='tf_efficientnet_b1_ns', n_class=5, pretrained=True), 224
    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    # input_s = (3, image_size, image_size)
    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))
    pass
