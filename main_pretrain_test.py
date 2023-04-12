import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch
from util.data_process import *
from util.metrics import get_MSE, get_MAE, get_MAPE

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_3', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')
    parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
    parser.add_argument('--random', type=str, default='random', help='fixed or random')
    parser.add_argument('--fraction', type=int, default=20, help='fraction')

    parser.add_argument('--mask_ratio',  type=float,
                        help='Masking ratio (percentage of removed patches).')
    # parser.add_argument('--base_channels', type=int,
    #                     default=128, help='number of feature maps')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of flow image channels')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='data/TaxiBJ/P1', type=str,
                        help='dataset path')
    # parser.add_argument('--data_path', default='data/BikeNYC', type=str,
    #                     help='dataset path')

    parser.add_argument('--scaler_X', type=int, default=1,
                        help='scaler of coarse-grained flows')
    parser.add_argument('--scaler_Y', type=int, default=1,
                        help='scaler of fine-grained flows')


    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2022, type=int)

    parser.add_argument('--resume', default='./Saved_model/data/TaxiBJ/P1/20/random-mae_vit_3-ps4/2022-04-27_16 13 19',
                        help='resume from checkpoint')
    parser.add_argument('--eval', default=True, help='test True')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    rmses = [np.inf]

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    random.seed(seed)

    cudnn.benchmark = True


    # load training set and validation set
    _, _, test_dataloader, sample_index = get_dataloader_inf2(
        args.data_path, args.scaler_X, args.scaler_Y, args.fraction, args.batch_size)


    # define the model
    model = models_mae.__dict__[args.model](patch_size=args.patch_size, in_chans=args.channels, norm_pix_loss=args.norm_pix_loss,
                                            img_size=args.input_size, random=args.random, sample_index=sample_index)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    args.resume = args.resume + '/checkpoint-best.pth'
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    model.eval()

    total_mse, total_mae, total_mape = 0, 0, 0
    for j, (flows_c, _, flows_f) in enumerate(test_dataloader):
        loss, preds = model(flows_c, flows_f, mask_ratio=args.mask_ratio)
        preds = preds.cpu().detach().numpy() * args.scaler_Y
        flows_f = flows_f.cpu().detach().numpy() * args.scaler_Y
        total_mse += get_MSE(preds, flows_f) * len(flows_c)
        total_mae += get_MAE(preds, flows_f) * len(flows_c)
        total_mape += get_MAPE(preds, flows_f) * len(flows_c)
    rmse = np.sqrt(total_mse / len(test_dataloader.dataset))
    mae = total_mae / len(test_dataloader.dataset)
    mape = total_mape / len(test_dataloader.dataset)

    print("RMSE\t{:.4f}\tMAE\t{:.4f}\tMAPE\t{:.4f}".format(rmse, mae, mape))
    f = open('{}/test_result.txt'.format(args.output_dir), 'a')
    f.write("RMSE\t{:.4f}\tMAE\t{:.4f}\tMAPE\t{:.4f}\n".format(rmse, mae, mape))
    f.close()



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = args.resume
    args.log_dir = args.output_dir
    args.mask_ratio = 1.0 - 1.0 / args.upscale_factor ** 2
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
