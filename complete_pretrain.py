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

from engine_pretrain import train_one_epoch_complete
from util.data_process import *
from util.metrics import get_MSE


def get_args_parser():
    parser = argparse.ArgumentParser('MAE complete_dat_pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
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
    # parser.add_argument('--random', type=str, default='fixed', help='fixed or random')

    parser.add_argument('--data_path', default='data/TaxiBJ/P1', type=str,
                        help='dataset path')
    parser.add_argument('--fraction', type=int, default=40, help='fraction')

    parser.add_argument('--len_closeness', type=int, default=2)
    parser.add_argument('--len_period', type=int, default=1)
    parser.add_argument('--len_trend', type=int, default=1)
    parser.add_argument('--T', type=int, default=28)

    parser.add_argument('--type', default='softmax', type=str, help='POI loss type')
    parser.add_argument('--margin', type=float, default=100, help='POI loss margin')
    parser.add_argument('--gama', type=float, default=1, help='POI loss')
    parser.add_argument('--alpha', type=float, default=10, help='Space loss')
    parser.add_argument('--beta', type=float, default=1, help='Time loss')
    parser.add_argument('--d_alpha', type=float, default=0, help='Decoder Space loss')
    parser.add_argument('--d_beta', type=float, default=0, help='Decoder Time loss')


    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--change_epoch', default=0, type=int, help='no requires_gred epoch')

    parser.add_argument('--scaler_X', type=int, default=1,
                        help='scaler of coarse-grained flows')
    parser.add_argument('--scaler_Y', type=int, default=1,
                        help='scaler of fine-grained flows')

    # parser.add_argument('--mask_ratio', type=float,
    #                     help='Masking ratio (percentage of removed patches).')
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
    parser.add_argument('--blr', type=float, default=0.01, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters

    # parser.add_argument('--data_path', default='data/BikeNYC', type=str,
    #                     help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2022, type=int)

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
    r_rmses = [np.inf]

    # misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    with open(os.path.join(args.output_dir, "args.txt"), mode="a", encoding="utf-8") as f:
        f.write("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # load training set and validation set
    train_dataloader, valid_dataloader, _, sample_index, mmn = get_dataloader_inf3(args.data_path, args.scaler_X,
        args.scaler_Y, args.fraction, args.batch_size, args.len_closeness, args.len_period, args.len_trend, args.T, 4)

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

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume != '':
        args.resume = args.resume + '/checkpoint-best.pth'
        # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.change_epoch != 0:
        for p in model.blocks.parameters():
            p.requires_grad = False
        for p in model.decoder_blocks.parameters():
            p.requires_grad = False

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.change_epoch != 0 and epoch == args.change_epoch:
            for p in model.blocks.parameters():
                p.requires_grad = True
            for p in model.decoder_blocks.parameters():
                p.requires_grad = True

        train_stats = train_one_epoch_complete(
            model, train_dataloader,
            optimizer, device, epoch, loss_scaler, mmn=mmn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            model.eval()

            total_mse, total_r_mse, total_space_loss, total_time_loss, total_mae = 0, 0, 0, 0, 0
            total_d_space_loss, total_d_time_loss, total_poi_loss = 0, 0, 0
            for j, data in enumerate(valid_dataloader):
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

            rmse = np.sqrt(total_mse / len(valid_dataloader.dataset))
            r_rmse = np.sqrt(total_r_mse / len(valid_dataloader.dataset))
            s_rmse = np.sqrt(total_space_loss / len(valid_dataloader.dataset))
            sd_rmse = np.sqrt(total_d_space_loss / len(valid_dataloader.dataset))
            t_rmse = np.sqrt(total_time_loss / len(valid_dataloader.dataset))
            td_rmse = np.sqrt(total_d_time_loss / len(valid_dataloader.dataset))
            poi_loss = total_poi_loss / len(valid_dataloader.dataset)

            if r_rmse < np.min(r_rmses):
                print("epoch\t{}\tRMSE\t{}\t".format(epoch, rmse))
                misc.save_model_c(
                    args=args, model=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best')
                f = open('{}/results.txt'.format(args.output_dir), 'a')
                f.write("\n\nBest: epoch\t{}\tRMSE\t{:.2f}\tR_RMSE\t{:.2f}\t".format(epoch, rmse,r_rmse))
                f.write("POI_loss\t{:.2f}\tSpace_RMSE\t{:.2f}\tD_Space_RMSE\t{:.2f}\tTime_RMSE\t{:.2f}\tD_Time_RMSE\t{:.2f}\n\n".format(poi_loss, s_rmse, sd_rmse, t_rmse, td_rmse))
                f.close()
            r_rmses.append(r_rmse)

            f = open('{}/results.txt'.format(args.output_dir), 'a')
            f.write("epoch\t{}\tRMSE\t{:.2f}\tR_RMSE\t{:.2f}\t".format(epoch, rmse, r_rmse))
            f.write("POI_loss\t{:.2f}\tSpace_RMSE\t{:.2f}\tD_Space_RMSE\t{:.2f}\tTime_RMSE\t{:.2f}\tD_Time_RMSE\t{:.2f}\n".format(poi_loss, s_rmse,
                                                                                                                sd_rmse,
                                                                                                                t_rmse,
                                                                                                                td_rmse))
            f.close()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    pre = ''
    if args.resume != None:
        pre = 'train-'
    args.output_dir = 'Saved_model/{}/{}/complete-pre-{}/{date:%Y-%m-%d_%H %M %S}'.format(args.data_path,
                                                                                          args.fraction,
                                                                                          args.model,
                                                                                          date=datetime.datetime.now())
    args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


