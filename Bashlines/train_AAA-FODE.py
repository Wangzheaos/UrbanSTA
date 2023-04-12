import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import timm
sys.path.append("..")
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae2, models_mae2_2, models_mae2_2_poi


from engine_pretrain import train_one_epoch
from util.data_process import *
from util.metrics import get_MSE, get_MAE, get_MAPE
from Models.FODE import FODE

import seaborn as sns
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('complete_pretrain_test', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=80, type=int)
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

    parser.add_argument('--base_channels', type=int,
                        default=128, help='number of feature maps')
    parser.add_argument('--sample_interval', type=int, default=20,
                        help='interval between validation')
    parser.add_argument('--harved_epoch', type=int, default=50,
                        help='halved at every x interval')

    parser.add_argument('--len_closeness', type=int, default=2)
    parser.add_argument('--len_period', type=int, default=1)
    parser.add_argument('--len_trend', type=int, default=1)
    parser.add_argument('--T', type=int, default=28)


    parser.add_argument('--fraction', type=int, default=20, help='fraction')
    parser.add_argument('--data_path', default='TaxiBJ/P1', type=str, help='dataset path')
    parser.add_argument('--upscale_factor', type=int, default=4
                        , help='upscale factor')
    parser.add_argument('--resume', default='../Saved_model/data/TaxiBJ/P1/20/complete-pre-mae_vit_1/2022-07-31_22 05 01',
                        help='resume from checkpoint')


    parser.add_argument('--type', default='softmax', type=str, help='POI loss type')
    parser.add_argument('--margin', type=float, default=100, help='POI loss margin')
    parser.add_argument('--gama', type=float, default=0, help='POI loss')
    parser.add_argument('--alpha', type=float, default=0, help='Space loss')
    parser.add_argument('--beta', type=float, default=0, help='Time loss')
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
    parser.add_argument('--blr', type=float, default=1e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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

    # path for saving model
    save_path = './Saved_model/{}/{}/{}/{}'.format('AAA-FODE', args.data_path, args.fraction, args.upscale_factor)
    os.makedirs(save_path, exist_ok=True)

    datapath = os.path.join('../data', args.data_path)
    # load training set and validation set
    train_dataloader, valid_dataloader, test_dataloader, sample_index, _ = get_dataloader_inf3(datapath, args.scaler_X,
                                                                              args.scaler_Y, args.fraction,
                                                                              args.batch_size, args.len_closeness,
                                                                              args.len_period, args.len_trend, args.T,
                                                                                               args.upscale_factor)
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

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # initial model
    F_model = FODE(in_channels=args.channels,
                 out_channels=args.channels,
                 base_channels=args.base_channels,
                upscale_factor=args.upscale_factor)
    torch.nn.utils.clip_grad_norm(F_model.parameters(), max_norm=5.0)
    criterion = nn.MSELoss()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if cuda:
        F_model.cuda()
        criterion.cuda()


    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(F_model, args.weight_decay)
    lr = 1e-3

    coptimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(
        F_model.parameters(), lr=lr, betas=(0.9, 0.999))

    print(optimizer)
    loss_scaler = NativeScaler()

    args.resume = args.resume + '/checkpoint-best.pth'
    misc.load_model_c(args=args, model=model, optimizer=coptimizer, loss_scaler=loss_scaler)

    model.eval()

    iter = 0
    rmses = [np.inf]
    maes = [np.inf]
    # for epoch in range(args.epochs):
    #     train_loss = 0
    #     for i, data in enumerate(train_dataloader):
    #         len_num = 0
    #         if args.len_closeness != 0:
    #             len_num += 1
    #         if args.len_period != 0:
    #             len_num += 1
    #         if args.len_trend != 0:
    #             len_num += 1
    #         if len_num == 1:
    #             flows_xc, flows_x, flows_y, flows_z = data
    #             flows_xc = flows_xc.to(device, non_blocking=True)
    #             flows_x = flows_x.to(device, non_blocking=True)
    #             flows_y = flows_y.to(device, non_blocking=True)
    #             closs, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_x, flows_y)
    #         elif len_num == 2:
    #             flows_xc, flows_xp, flows_x, flows_y, flows_z = data
    #             flows_xc = flows_xc.to(device, non_blocking=True)
    #             flows_xp = flows_xp.to(device, non_blocking=True)
    #             flows_x = flows_x.to(device, non_blocking=True)
    #             flows_y = flows_y.to(device, non_blocking=True)
    #             closs, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp, flows_x, flows_y)
    #         else:
    #             flows_xc, flows_xp, flows_xt, flows_x, flows_y, flows_z = data
    #             flows_xc = flows_xc.to(device, non_blocking=True)
    #             flows_xp = flows_xp.to(device, non_blocking=True)
    #             flows_xt = flows_xt.to(device, non_blocking=True)
    #             flows_x = flows_x.to(device, non_blocking=True)
    #             flows_y = flows_y.to(device, non_blocking=True)
    #             flows_z = flows_z.to(device, non_blocking=True)
    #             closs, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp, flows_xt, flows_x, flows_y)
    #
    #         F_model.train()
    #         optimizer.zero_grad()
    #
    #         # generate images with high resolution
    #         gen_hr = F_model(preds)
    #         loss = criterion(gen_hr, flows_z)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f] [c_loss: %f]" % (epoch, args.epochs, i, len(train_dataloader),
    #                                                         np.sqrt(loss.item()), np.sqrt(closs.item())))
    #
    #         # counting training mse
    #         train_loss += loss.item() * len(preds)
    #
    #         iter += 1
    #         # validation phase
    #         if iter % args.sample_interval == 0:
    #             F_model.eval()
    #             total_mse, total_mae = 0, 0
    #
    #             for j, data in enumerate(valid_dataloader):
    #                 len_num = 0
    #                 if args.len_closeness != 0:
    #                     len_num += 1
    #                 if args.len_period != 0:
    #                     len_num += 1
    #                 if args.len_trend != 0:
    #                     len_num += 1
    #                 if len_num == 1:
    #                     flows_xc, flows_x, flows_y, flows_z = data
    #                     flows_xc = flows_xc.to(device, non_blocking=True)
    #                     flows_x = flows_x.to(device, non_blocking=True)
    #                     flows_y = flows_y.to(device, non_blocking=True)
    #                     loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_x,
    #                                                                                                  flows_y)
    #                 elif len_num == 2:
    #                     flows_xc, flows_xp, flows_x, flows_y, flows_z = data
    #                     flows_xc = flows_xc.to(device, non_blocking=True)
    #                     flows_xp = flows_xp.to(device, non_blocking=True)
    #                     flows_x = flows_x.to(device, non_blocking=True)
    #                     flows_y = flows_y.to(device, non_blocking=True)
    #                     loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp,
    #                                                                                                  flows_x, flows_y)
    #                 else:
    #                     flows_xc, flows_xp, flows_xt, flows_x, flows_y, flows_z = data
    #                     flows_xc = flows_xc.to(device, non_blocking=True)
    #                     flows_xp = flows_xp.to(device, non_blocking=True)
    #                     flows_xt = flows_xt.to(device, non_blocking=True)
    #                     flows_x = flows_x.to(device, non_blocking=True)
    #                     flows_y = flows_y.to(device, non_blocking=True)
    #                     flows_z = flows_z.to(device, non_blocking=True)
    #                     loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp,
    #                                                                                                  flows_xt, flows_x,
    #                                                                                                  flows_y)
    #
    #                 preds = F_model(preds)
    #                 preds = preds.cpu().detach().numpy()
    #                 flows_z = flows_z.cpu().detach().numpy()
    #                 total_mse += get_MSE(preds, flows_z) * len(preds)
    #             rmse = np.sqrt(total_mse / len(valid_dataloader.dataset))
    #             if rmse < np.min(rmses):
    #                 print("iter\t{}\tRMSE\t{:.6f}\t".format(iter, rmse))
    #
    #                 print(save_path)
    #                 torch.save(F_model.state_dict(),
    #                        '{}/final_model.pt'.format(save_path))
    #                 f = open('{}/results.txt'.format(save_path), 'a')
    #                 f.write("epoch\t{}\titer\t{}\tRMSE\t{:.6f}\n".format(epoch, iter, rmse))
    #                 f.close()
    #             rmses.append(rmse)
    #
    #     # halve the learning rate
    #     if epoch % args.harved_epoch == 0 and epoch != 0:
    #         lr /= 2
    #         optimizer = torch.optim.Adam(
    #             F_model.parameters(), lr=lr, betas=(0.9, 0.999))
    #         f = open('{}/results.txt'.format(save_path), 'a')
    #         f.write("half the learning rate!\n")
    #         f.close()
    #


    F_model.load_state_dict(torch.load('{}/final_model.pt'.format(save_path)))
    F_model.eval()
    total_mse, total_mae, total_mape = 0, 0, 0

    for j, data in enumerate(test_dataloader):
        len_num = 0
        if args.len_closeness != 0:
            len_num += 1
        if args.len_period != 0:
            len_num += 1
        if args.len_trend != 0:
            len_num += 1
        if len_num == 1:
            flows_xc, flows_x, flows_y, flows_z = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_x,
                                                                                         flows_y)
        elif len_num == 2:
            flows_xc, flows_xp, flows_x, flows_y, flows_z = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp,
                                                                                         flows_x, flows_y)
        else:
            flows_xc, flows_xp, flows_xt, flows_x, flows_y, flows_z = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_xt = flows_xt.to(device, non_blocking=True)
            flows_x = flows_x.to(device, non_blocking=True)
            flows_y = flows_y.to(device, non_blocking=True)
            flows_z = flows_z.to(device, non_blocking=True)
            loss, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, preds, _ = model(flows_xc, flows_xp,
                                                                                         flows_xt, flows_x,
                                                                                         flows_y)

        preds = F_model(preds)
        preds = preds.cpu().detach().numpy()
        flows_z = flows_z.cpu().detach().numpy()


        if j == 5:
            plt.figure(figsize=(6.2, 6))
            plt.axis('off')
            ax = sns.heatmap(preds[30][0], cmap="RdYlGn_r", cbar=False)
            plt.savefig('../Figure/AAA-FODE.png', bbox_inches='tight')
            plt.show()


        total_mse += get_MSE(preds, flows_z) * len(flows_x)
        total_mae += get_MAE(preds, flows_z) * len(flows_x)
        total_mape += get_MAPE(preds, flows_z) * len(flows_x)
    rmse = np.sqrt(total_mse / len(test_dataloader.dataset))
    mse = total_mse / len(test_dataloader.dataset)
    mae = total_mae / len(test_dataloader.dataset)
    mape = total_mape / len(test_dataloader.dataset)

    with open('{}/test_results.txt'.format(save_path), 'w') as f:
        f.write("RMSE\t{:.2f}\tMAE\t{:.2f}\tMAPE\t{:.4f}\n".format(rmse, mae, mape))
    print('Test RMSE = {:.2f}\nMAE = {:.2f}\nMAPE = {:.4f}'.format(rmse, mae, mape))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
