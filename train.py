import os
import os.path as osp
import time
import math
import wandb

from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

import numpy as np
import random

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/total'))
                        # default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/total/ufo/train.json'))
                        # default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/Upstage'))
                        # default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/ICDAR17_Korean'))

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--optim', default='adam')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, optim):
    train_dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset = SceneTextDataset(data_dir, split='val', image_size=image_size, crop_size=input_size)
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    if optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    wandb.watch(model)

    train_step, val_step = 0, 0
    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        # cls_train, angle_train, iou_train = 0, 0, 0
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train = loss.item()
                epoch_loss += loss_train

                pbar.update(1)
                train_dict = {
                    '[Train] Cls loss': extra_info['cls_loss'], '[Train] Angle loss': extra_info['angle_loss'],
                    '[Train] IoU loss': extra_info['iou_loss'], '[Train] Cur loss': loss_train
                }
                pbar.set_postfix(train_dict)
                wandb.log(train_dict)
                # wandb.log(train_dict, step=train_step)
                train_step += 1        

        scheduler.step()
        # wandb.log({"Train": train_dict})

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        wandb.log({"[Train] Mean loss": epoch_loss / train_num_batches})#, step=epoch)


        model.eval()
        val_loss = 0
        # cls_val, angle_val, iou_val = 0, 0, 0
        with tqdm(total=val_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                pbar.set_description('[Val Epoch {}]'.format(epoch + 1))
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                loss_val = loss.item()
                val_loss += loss_val
                # cls_val += extra_info['cls_loss']
                # angle_val += extra_info['angle_loss']
                # iou_val += extra_info['iou_loss']
                
                pbar.update(1)

                val_dict = {
                    '[Val] Cls loss': extra_info['cls_loss'], '[Val] Angle loss': extra_info['angle_loss'],
                    '[Val] IoU loss': extra_info['iou_loss'], '[Val] Cur loss': loss_val
                }
                pbar.set_postfix(val_dict)
                wandb.log(val_dict)
                # wandb.log(val_dict, step=val_step)
                val_step += 1

        wandb.log({"[Val] Mean loss": val_loss / val_num_batches})#, step=epoch)


        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
                
            ckpt_fpath = osp.join(model_dir, f'{epoch+1}_epoch.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def set_random_seed(random_seed):
    torch.manual_seed(random_seed) # torch 
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True # cudnn
    torch.backends.cudnn.benchmark = False # cudnn
    np.random.seed(random_seed) # numpy
    random.seed(random_seed) # random


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    set_random_seed(1)
    args = parse_args()
    wandb.init(project="OCR")
    #wandb.run.name = 'Upstage'
    wandb.config.update(args)
    main(args)