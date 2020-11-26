import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import *
import time
from networks.models import get_efficientnet
from datasets.dataset import CLDCDataset, get_train_transforms, get_valid_transforms
from losses.losses import LabelSmoothing, FocalLoss
from catalyst.data.sampler import BalanceClassSampler
from utils.utils import AverageMeter, calculate_metrics, Logger


def eval_model(epoch, is_save=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_score = AverageMeter()
    model.eval()
    num_steps = len(eval_loader)
    print(f'total batches: {num_steps}')
    end = time.time()
    eval_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (XI, label) in enumerate(eval_loader):
            x = Variable(XI.cuda(device_id))
            # label = Variable(torch.LongTensor(label).cuda(device_id))
            label = Variable(label.cuda(device_id))

            # Forward pass: Compute predicted y by passing x to the model
            output = model(x)
            # Compute and print loss
            loss = eval_criterion(output, label)
            losses.update(loss.data.item(), x.size(0))
            # update metrics
            output = nn.Softmax(dim=1)(output)
            confs, predicts = torch.max(output.detach(), dim=1)
            acc_score.update(calculate_metrics(predicts.cpu(), label.cpu()), 1)

            lr = optimizer.param_groups[0]['lr']
            batch_time.update(time.time() - end)
            end = time.time()

            if i % LOG_FREQ == 0:
                print(f'{epoch} [{i}/{num_steps}]\t'
                      f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'acc {acc_score.val:.4f} ({acc_score.avg:.4f})\t'
                      f'lr {lr:.8f}')

    print(f' *  Eval loss {losses.avg:.4f}\t'f'acc({acc_score.avg:.4f})')
    if is_save:
        train_logger.log(phase="eval", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(acc_score.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    return acc_score.avg, losses.avg

def train_model(epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_score = AverageMeter()
    model.train()
    num_steps = len(train_loader)
    print(f'total batches: {num_steps}')
    end = time.time()

    for i, (XI, label) in enumerate(train_loader):
        x = Variable(XI.cuda(device_id))
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        label = Variable(label.cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        output = model(x)
        # Compute and print loss
        loss = criterion(output, label)
        # update metrics
        losses.update(loss.data.item(), x.size(0))
        confs, predicts = torch.max(output.detach(), dim=1)
        acc_score.update(calculate_metrics(predicts.cpu(), label.cpu()), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                  f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'acc {acc_score.val:.4f} ({acc_score.avg:.4f})\t'
                  f'lr {lr:.8f}')

    print(f' *  Train loss {losses.avg:.4f}\t'f'acc({acc_score.avg:.4f})')
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(acc_score.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    # scheduler.step()
    return losses.val


if __name__ == '__main__':
    k = 1
    LOG_FREQ = 50
    batch_size = 64
    test_batch_size = 128
    device_id = 2
    lr = 0.001
    epoch_start = 1
    num_epochs = epoch_start + 100
    model_name = 'efficientnet-b1'
    writeFile = './output/logs/' + model_name + '_' + str(k)
    store_name = './output/weights/' + model_name + '_' + str(k)
    if not os.path.isdir(store_name):
        os.makedirs(store_name)
    model_path = None
    # model_path = '/data1'
    model = get_efficientnet(model_name=model_name)
    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)
    train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    # criterion = FocalLoss(gamma=2)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=2)

    is_train = True
    if is_train:
        xdl = CLDCDataset(is_one_hot=True, transform=get_train_transforms(size=300, is_alb=False),
                          is_alb=False, k=k)
        # train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=False, num_workers=4,
        #                           sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="upsampling"))
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=True, num_workers=4)
        train_dataset_len = len(xdl)
        xdl_eval = CLDCDataset(data_type='val', transform=get_valid_transforms(size=300, is_alb=False),
                               is_alb=False, is_one_hot=False, k=k)
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)
        max_acc = 0 if epoch_start == 1 else eval_model(epoch=epoch_start, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            train_model(epoch)
            acc, val_loss = eval_model(epoch)
            scheduler.step(val_loss)
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), '{}/e{}_acc_{:.4f}.pth'.format(store_name, epoch, acc))
            print('Current max acc:', max_acc)
        torch.save(model.state_dict(), '{}/e{}_acc_{:.4f}.pth'.format(store_name, 'last_50', acc))

    else:
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = CLDCDataset(data_type='val', is_one_hot=False, transform=get_valid_transforms(size=224))
        eval_loader = DataLoader(xdl_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        eval_model(epoch=0, is_save=False)
        print('Total time:', time.time() - start)






