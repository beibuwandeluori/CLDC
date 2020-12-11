from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch
from torch import nn
import os
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
# from torch.cuda.amp import autocast, GradScaler

from catalyst.data.sampler import BalanceClassSampler
from datasets.dataset_v2 import CassavaDataset, get_train_transforms, get_valid_transforms
from losses.losses import LabelSmoothing
from networks.models import CassvaImgClassifier
from utils.utils import AverageMeter, calculate_metrics, Logger

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device):
    model.train()
    losses = AverageMeter()
    acc_score = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        # print(imgs.shape, image_labels.shape, image_preds.shape)
        loss = loss_fn(image_preds, image_labels)

        losses.update(loss.data.item(), imgs.size(0))
        # confs, predicts = torch.max(imgs.detach(), dim=1)
        # acc_score.update(calculate_metrics(predicts.cpu(), image_labels.cpu()), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {loss:.4f}'
            pbar.set_description(description)

    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(0.0, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, is_save=True):
    model.eval()
    losses = AverageMeter()
    acc_score = AverageMeter()

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        with torch.no_grad():
            imgs = imgs.to(device).float()
            image_labels = image_labels.to(device).long()

            image_preds = model(imgs)

            loss = loss_fn(image_preds, image_labels)
            losses.update(loss.data.item(), imgs.size(0))

            output = nn.Softmax(dim=1)(image_preds)
            confs, predicts = torch.max(output.detach(), dim=1)
            acc_score.update(calculate_metrics(predicts.cpu(), image_labels.cpu()), imgs.size(0))

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
                description = f'epoch {epoch} loss: {losses.avg:.4f} acc: {acc_score.avg:.4f}'
                pbar.set_description(description)

    print('validation multi-class accuracy = {0:.4f} loss{1:.4f}'.format(acc_score.avg, losses.avg))
    if is_save:
        train_logger.log(phase="eval", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(acc_score.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    return acc_score.avg, losses.avg


def prepare_dataloader(df, trn_idx, val_idx, do_fmix=False, do_cutmix=False,
                       data_root='/raid/chenby/cassava-leaf-disease-classification/train_images'):

    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True,
                              one_hot_label=True, do_fmix=do_fmix, do_cutmix=do_cutmix)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True,
                              one_hot_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b5_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 16,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:3'
}

if __name__ == '__main__':
    store_name = './output_v2/weights/' + CFG['model_arch'] + '_' + str(CFG['img_size']) + '_cutmix'
    if not os.path.isdir(store_name):
        os.makedirs(store_name)
    train = pd.read_csv('/raid/chenby/cassava-leaf-disease-classification/train.csv')
    # for training only, need nightly build pytorch

    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True,
                            random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        writeFile = './output_v2/logs/' + CFG['model_arch'] + '_' + str(fold) + '_' + str(CFG['img_size']) + '_cutmix'
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])

        print('Training with {} started'.format(fold))
        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, do_cutmix=True, do_fmix=False)
        device = torch.device(CFG['device'])

        model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
        # scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1,
                                                                         eta_min=CFG['min_lr'], last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

        # loss_tr = nn.CrossEntropyLoss().to(device)  # MyCrossEntropyLoss().to(device)
        # loss_fn = nn.CrossEntropyLoss().to(device)

        loss_tr = LabelSmoothing(smoothing=0.05).to(device)
        loss_fn = LabelSmoothing(smoothing=0.05).to(device)

        best_acc = 0.
        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device)
            val_acc, val_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device)
            # scheduler.step(val_loss)
            scheduler.step()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), '{}/fold_{}_{}_{:.4f}'.
                           format(store_name, fold, epoch, val_acc))
            print('Current acc:{:.4f}, Best acc:{:.4f}'.format(val_acc, best_acc))
        del model, optimizer, train_loader, val_loader, scheduler

        torch.cuda.empty_cache()
