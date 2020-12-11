import tensorboardX
from sklearn.metrics import log_loss, accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score
import torch

class Logger(object):
    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_metrics(outputs, targets):
    # print(len(outputs.detach().numpy().shape), len(targets.data.numpy().shape))
    if len(targets.data.numpy().shape) > 1:
        _, targets = torch.max(targets.detach(), dim=1)
    acc = accuracy_score(outputs.detach().numpy(), targets.detach().numpy())
    # loss = log_loss(outputs.detach().numpy(), targets.data.numpy())
    return acc

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _ , pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])