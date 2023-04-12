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

import models_mae2, models_mae2_2, models_mae2_2_poi

from engine_pretrain import train_one_epoch
from util.data_process import *
from util.metrics import get_MSE, get_MAE, get_MAPE


def get_args_parser():
    parser = argparse.ArgumentParser('complete_pretrain_test', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--MModel', default='models_mae2_2_poi', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model', default='mae_vit_1', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--input_size', default=8, type=int,
                        help='images input size')
    # parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
    # parser.add_argument('--random', type=str, default='random', help='fixed or random')
    parser.add_argument('--len_closeness', type=int, default=2)
    parser.add_argument('--len_period', type=int, default=1)
    parser.add_argument('--len_trend', type=int, default=1)
    parser.add_argument('--T', type=int, default=28)

    parser.add_argument('--fraction', type=int, default=60, help='fraction')
    parser.add_argument('--resume', default='./Saved_model/data/TaxiBJ/P1/60/complete-pre-mae_vit_1/2022-07-29_15 01 20',
                        help='resume from checkpoint')

    parser.add_argument('--type', default='softmax', type=str, help='POI loss type')
    parser.add_argument('--margin', type=float, default=100, help='POI loss margin')
    parser.add_argument('--gama', type=float, default=10, help='POI loss')
    parser.add_argument('--alpha', type=float, default=10, help='Space loss')
    parser.add_argument('--beta', type=float, default=10, help='Time loss')
    parser.add_argument('--d_alpha', type=float, default=0, help='Decoder Space loss')
    parser.add_argument('--d_beta', type=float, default=0, help='Decoder Time loss')

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
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
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
    _, _, test_dataloader, sample_index, _ = get_dataloader_inf3(args.data_path, args.scaler_X,
                                                                              args.scaler_Y, args.fraction,
                                                                              args.batch_size, args.len_closeness,
                                                                              args.len_period, args.len_trend, args.T,4)
    tsum = args.len_closeness + args.len_period + args.len_trend
    # define the model
    if args.MModel == 'models_mae2_2':
        model = models_mae2_2.__dict__[args.model](patch_size=args.patch_size, in_chans=args.channels,
                                            img_size=args.input_size, sample_index=sample_index, T_len=tsum)
    elif args.MModel == 'models_mae2':
        model = models_mae2.__dict__[args.model](patch_size=args.patch_size, in_chans=args.channels,
                                                   img_size=args.input_size, sample_index=sample_index, T_len=tsum)

    elif args.MModel == 'models_mae2_2_poi':
        model = models_mae2_2_poi.__dict__[args.model](patch_size=args.patch_size, in_chans=args.channels, margin=args.margin,
                                                   type=args.type, img_size=args.input_size, sample_index=sample_index, T_len=tsum)

    else:
        print("model name wrong!")
        return 0

    model.to(device)


    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    args.resume = args.resume + '/checkpoint-best.pth'
    misc.load_model_c(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    model.eval()

    total_mse, total_r_mse, total_space_loss, total_time_loss, total_mae, total_mape = 0, 0, 0, 0, 0, 0
    total_d_space_loss, total_d_time_loss, total_poi_loss = 0, 0, 0
    for j, data in enumerate(test_dataloader):
        len_num = 0
        if args.len_closeness != 0:
            len_num += 1
        if args.len_period != 0:
            len_num += 1
        if args.len_trend != 0:
            len_num += 1
        if len_num == 1:
            flows_xc, flows_x, flows_y, _ = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_x, flows_y)
        elif len_num == 2:
            flows_xc, flows_xp, flows_x, flows_y, _ = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp, flows_x, flows_y)
        else:
            flows_xc, flows_xp, flows_xt, flows_x, flows_y, _ = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_xt = flows_xt.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp, flows_xt, flows_x, flows_y)

        Loss = all_loss + args.alpha * s_loss + args.beta * t_loss + args.d_alpha * sd_loss + args.d_beta * td_loss + args.gama * poi_loss
        total_space_loss += s_loss.cpu().detach().numpy() * len(flows_x)
        total_d_space_loss += sd_loss.cpu().detach().numpy() * len(flows_x)
        total_time_loss += t_loss.cpu().detach().numpy() * len(flows_x)
        total_d_time_loss += td_loss.cpu().detach().numpy() * len(flows_x)
        total_r_mse += loss.cpu().detach().numpy() * len(flows_x)
        total_poi_loss += poi_loss.cpu().detach().numpy() * len(flows_x)

        total_mse += Loss.cpu().detach().numpy() * len(flows_x)

        n, c, h, w = preds.shape
        mask_preds = torch.zeros(n, c, len(sample_index))
        mask_flows_y = torch.zeros(n, c, len(sample_index))
        for i in range(len(sample_index)):
            p = int(sample_index[i] / w)
            q = int(sample_index[i] % w)
            mask_preds[:, :, i] = preds[:, :, p, q]
            mask_flows_y[:, :, i] = flows_y[:, :, p, q]

        total_mae += get_MAE(mask_preds.cpu().detach().numpy(), mask_flows_y.cpu().detach().numpy()) * len(flows_x)
        total_mape += get_MAPE(mask_preds.cpu().detach().numpy(), mask_flows_y.cpu().detach().numpy()) * len(flows_x)

    rmse = np.sqrt(total_mse / len(test_dataloader.dataset)) * args.scaler_Y
    r_rmse = np.sqrt(total_r_mse / len(test_dataloader.dataset)) * args.scaler_Y
    s_rmse = np.sqrt(total_space_loss / len(test_dataloader.dataset)) * args.scaler_Y
    sd_rmse = np.sqrt(total_d_space_loss / len(test_dataloader.dataset)) * args.scaler_Y
    t_rmse = np.sqrt(total_time_loss / len(test_dataloader.dataset)) * args.scaler_Y
    td_rmse = np.sqrt(total_d_time_loss / len(test_dataloader.dataset)) * args.scaler_Y
    poi_loss = total_poi_loss / len(test_dataloader.dataset) * 1

    mae = total_mae / len(test_dataloader.dataset)  * args.scaler_Y
    mape = total_mape / len(test_dataloader.dataset) * args.scaler_Y

    print("RMSE\t{:.2f}\tR_RMSE\t{:.2f}\tMAE\t{:.2f}\tMAPE\t{:.4f}".format(rmse, r_rmse, mae, mape))
    print("POI_loss\t{:.2f}\tSpace_RMSE\t{:.2f}\tD_Space_RMSE\t{:.2f}\tTime_RMSE\t{:.2f}\tD_Time_RMSE\t{:.2f}\n".format(poi_loss, s_rmse, sd_rmse, t_rmse, td_rmse))

    f = open('{}/test_result.txt'.format(args.output_dir), 'a')
    f.write("RMSE\t{:.2f}\tR_RMSE\t{:.2f}\tMAE\t{:.2f}\tMAPE\t{:.4f}\n".format(rmse, r_rmse, mae, mape))
    f.write("POI_loss\t{:.2f}\tSpace_RMSE\t{:.2f}\tD_Space_RMSE\t{:.2f}\tTime_RMSE\t{:.2f}\tD_Time_RMSE\t{:.2f}\n\n".format(poi_loss, s_rmse, sd_rmse, t_rmse, td_rmse))

    f.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = args.resume
    args.log_dir = args.output_dir

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
