import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import random

from matplotlib.ticker import MultipleLocator

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import models_mae2_2
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae, models_mae_all, models_mae_all_fine_grained, models_mae2, models_mae2_2_poi, models_mae_poi

from engine_pretrain import train_one_epoch
from util.data_process import *
from util.metrics import get_MSE, get_MAE, get_MAPE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=50, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--CModel', default='models_mae2_2_poi', type=str, metavar='MODEL',
                        help='C model Name of model to train')
    parser.add_argument('--FModel', default='models_mae_all_fine_grained', type=str, metavar='MODEL',
                        help='F model Name of model to train')
    parser.add_argument('--c_model', default='mae_vit_1', type=str, metavar='MODEL',
                        help='Name of model to train complete module')
    parser.add_argument('--f_model', default='mae_vit_3', type=str, metavar='MODEL',
                        help='Name of model to train fine-grain module')
    parser.add_argument('--patch_size', default=2, type=int)
    parser.add_argument('--input_size', default=8, type=int,
                        help='images input size')
    parser.add_argument('--upscale_factor', type=int, default=2, help='upscale factor')
    parser.add_argument('--random', type=str, default='fixed', help='fixed or random')
    parser.add_argument('--fraction', type=int, default=20, help='fraction')
    parser.add_argument('--data_path', default='data/TaxiBJ/P4', type=str,
                        help='dataset path')
    parser.add_argument('--resume', default='./Saved_model/data/TaxiBJ/P4/20/main_fixed-mae_vit_3-ps2/2/2022-08-12_13 04 18',
                        help='resume from complete data model checkpoint')

    parser.add_argument('--len_closeness', type=int, default=2)
    parser.add_argument('--len_period', type=int, default=1)
    parser.add_argument('--len_trend', type=int, default=1)
    parser.add_argument('--T', type=int, default=28)

    parser.add_argument('--type', default='softmax', type=str, help='POI loss type')
    parser.add_argument('--margin', type=float, default=100, help='complete data POI loss margin')
    parser.add_argument('--fmargin', type=float, default=1, help='fine grained POI loss margin')

    parser.add_argument('--gama', type=float, default=0, help='complete data POI loss')
    parser.add_argument('--theta', type=float, default=1, help='fine grained POI loss')

    parser.add_argument('--mu', type=float, default=0.01, help='complete data loss')
    parser.add_argument('--nu', type=float, default=100, help='fine-grained loss')


    parser.add_argument('--alpha', type=float, default=10, help='complete data Space loss')
    parser.add_argument('--beta', type=float, default=10, help='complete data Time loss')
    parser.add_argument('--d_alpha', type=float, default=0, help='complete data Decoder Space loss')
    parser.add_argument('--d_beta', type=float, default=0, help='complete data Decoder Time loss')
    parser.add_argument('--delte', type=float, default=0, help='fine grained Space loss')


    parser.add_argument('--eval', default=True, help='test True')
    # parser.add_argument('--change_epoch', default=50, type=int, help='no requires_gred epoch')
    parser.add_argument('--scaler_X', type=int, default=1,
                        help='scaler of coarse-grained flows')
    parser.add_argument('--scaler_Y', type=int, default=1,
                        help='scaler of fine-grained flows')

    parser.add_argument('--mask_ratio', type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--channels', type=int, default=32,
                        help='number of flow image channels')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--clr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--flr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters


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
    train_dataloader, valid_dataloader, test_dataloader, sample_index, _ = get_dataloader_inf3(args.data_path, args.scaler_X,
                                                                              args.scaler_Y, args.fraction,
                                                                              args.batch_size, args.len_closeness,
                                                                              args.len_period, args.len_trend, args.T,
                                                                                               args.upscale_factor)

    # define the model
    tsum = args.len_closeness + args.len_period + args.len_trend

    if args.CModel == 'models_mae2_2':
        c_model = models_mae2_2.__dict__[args.c_model](patch_size=1, in_chans=1,
                                            img_size=8, sample_index=sample_index, T_len=tsum)
    elif args.CModel == 'models_mae2':
        c_model = models_mae2.__dict__[args.c_model](patch_size=1, in_chans=1,
                                                   img_size=8, sample_index=sample_index, T_len=tsum)
    elif args.CModel == 'models_mae2_2_poi':
        c_model = models_mae2_2_poi.__dict__[args.c_model](patch_size=1, in_chans=1, margin=args.margin,
                                        type=args.type, img_size=8, sample_index=sample_index, T_len=tsum)
    else:
        print("c model name wrong!")
        return 0

    if args.FModel == 'models_mae_poi':
        f_model = models_mae_poi.__dict__[args.f_model](patch_size=args.patch_size, in_chans=args.channels, fmargin=args.fmargin,
                                                    img_size=args.input_size, random=args.random,
                                                    sample_index=sample_index)
    elif args.FModel == 'models_mae':
        f_model = models_mae.__dict__[args.f_model](patch_size=args.patch_size, in_chans=args.channels,
                                                    img_size=args.input_size, random=args.random,
                                                    sample_index=sample_index)
    elif args.FModel == 'models_mae_all':
        f_model = models_mae_all.__dict__[args.f_model](patch_size=args.patch_size, in_chans=args.channels,
                                                    img_size=args.input_size, random=args.random,
                                                    sample_index=sample_index)
    elif args.FModel == 'models_mae_all_fine_grained':
        f_model = models_mae_all_fine_grained.__dict__[args.f_model](patch_size=args.patch_size, in_chans=args.channels,
                                                        img_size=args.input_size, random=args.random,
                                                        upscale_factor=args.upscale_factor,
                                                        mask_ratio=args.mask_ratio, sample_index=sample_index)
    else:
        print("f model name wrong!")
        return 0

    c_model.to(device)
    f_model.to(device)

    # following timm: set wd as 0 for bias and norm layers

    optimizer = torch.optim.AdamW([
        {'params': c_model.parameters(), 'lr': args.clr, 'betas': (0.9, 0.95)},
        {'params': f_model.parameters(), 'lr': args.flr, 'betas': (0.9, 0.95)}
    ])
    loss_scaler = NativeScaler()

    args.resume = args.resume + '/checkpoint-best.pth'
    misc.load_model(args=args, c_model=c_model, f_model=f_model, optimizer=optimizer, loss_scaler=loss_scaler)

    c_model.eval()
    f_model.eval()

    total_mse, total_mae, total_mape, total_loss1 = 0, 0, 0, 0
    total_d_space_loss, total_d_time_loss = 0, 0
    total_space_loss, total_time_loss = 0, 0
    total_sf_loss, total_poi_loss, total_poif_loss = 0, 0, 0

   #  print(len(test_dataloader.dataset))
   #  sys.exit(-1)

    for j, data in enumerate(test_dataloader):
        len_num = 0
        if args.len_closeness != 0:
            len_num += 1
        if args.len_period != 0:
            len_num += 1
        if args.len_trend != 0:
            len_num += 1
        if len_num == 1:
            flows_xc, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, latent = c_model(flows_xc, flows_c, flows_d)

        elif len_num == 2:
            flows_xc, flows_xp, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, latent = c_model(flows_xc, flows_xp, flows_c, flows_d)
        else:
            flows_xc, flows_xp, flows_xt, flows_c, flows_d, flows_f = data
            flows_xc = flows_xc.to(device, non_blocking=True)
            flows_xp = flows_xp.to(device, non_blocking=True)
            flows_xt = flows_xt.to(device, non_blocking=True)
            flows_c = flows_c.to(device, non_blocking=True)
            flows_d = flows_d.to(device, non_blocking=True)
            loss1, all_loss, poi_loss, s_loss, t_loss, sd_loss, td_loss, data, latent = c_model(flows_xc, flows_xp, flows_xt, flows_c, flows_d)

        data = data / args.scaler_X
        flows_f = flows_f / args.scaler_Y

        flows_f = flows_f.to(device, non_blocking=True)
        loss2, poif_loss, sf_loss, preds = f_model(data, latent, flows_f, mask_ratio=args.mask_ratio)

        print(j)

        # flows_xc: [250, 2, 1, 8, 8]
        # 可视化Divided Space-Time Attention
        # for i in range(flows_xc.shape[1]):
        #     plt.figure(figsize=(6.2, 6))
        #     plt.axis('off')
        #     ax = sns.heatmap(flows_xc[30][i][0].cpu().detach().numpy(), cmap="RdYlGn_r", mask=(flows_xc[30][i][0].cpu().detach().numpy() < 0), cbar=False)
        #     plt.savefig('Figure/Divided/flows_xc[' + str(i) +  '].png', bbox_inches='tight')
        #     plt.show()
        #     print(data[30][i][0])


        # 可视化数据补全和超分辨率过程
        # if j == 2:
        #     plt.figure(figsize=(6.2, 6))
        #     plt.axis('off')
        #     ax = sns.heatmap(flows_c[30][0].cpu().detach().numpy(), cmap="RdYlGn_r", mask=(flows_c[30][0].cpu().detach().numpy() < 0), cbar=False)
        #     plt.savefig('Figure/input.png', bbox_inches='tight')
        #     plt.show()
        #     # print(data[30][0])
        #
        #     plt.figure(figsize=(6.2, 6))
        #     plt.axis('off')
        #     ax = sns.heatmap(data[30][0].cpu().detach().numpy(), cmap="RdYlGn_r", cbar=False)
        #     plt.savefig('Figure/AAA-data complete.png', bbox_inches='tight')
        #     plt.show()
        #
        #     plt.figure(figsize=(6.2, 6))
        #     plt.axis('off')
        #     ax = sns.heatmap(flows_d[30][0].cpu().detach().numpy(), cmap="RdYlGn_r", cbar=False)
        #     plt.savefig('Figure/ground truth complete.png', bbox_inches='tight')
        #     plt.show()
        #

        for k in range(16):
            plt.figure(figsize=(8, 6))
            plt.axis('off')
            ax = sns.heatmap(preds[k][0].cpu().detach().numpy(), cmap="YlGnBu", cbar=True)
            plt.savefig('Figure/Complete_TaxiBJ-P4-2/BBB-pred resolution_' + str(k) + '.png', bbox_inches='tight')
            plt.show()


        # plt.figure(figsize=(8, 6))
        # plt.axis('off')
        # ax = sns.heatmap(flows_f[10][0].cpu().detach().numpy(), cmap="YlGnBu", cbar=True)
        # plt.savefig('Figure/Complete_TaxiBJ-P4-2/ground truth resolution.png', bbox_inches='tight')
        # plt.show()

        sys.exit(-1)

        preds = preds.cpu().detach().numpy() * args.scaler_Y
        flows_f = flows_f.cpu().detach().numpy() * args.scaler_Y
        total_mse += get_MSE(preds, flows_f) * len(flows_c)
        total_mae += get_MAE(preds, flows_f) * len(flows_c)
        total_mape += get_MAPE(preds, flows_f) * len(flows_c)


        # 数据曲线图可视化
        # if j == 0:
        #     Picture_data(j, flows_f, preds)


        total_loss1 += loss1.cpu().detach().numpy() * len(flows_c)
        total_space_loss += s_loss.cpu().detach().numpy() * len(flows_c)
        total_d_space_loss += sd_loss.cpu().detach().numpy() * len(flows_c)
        total_time_loss += t_loss.cpu().detach().numpy() * len(flows_c)
        total_d_time_loss += td_loss.cpu().detach().numpy() * len(flows_c)
        total_sf_loss += sf_loss.cpu().detach().numpy() * len(flows_c)
        total_poi_loss += poi_loss.cpu().detach().numpy() * len(flows_c)
        total_poif_loss += poif_loss.cpu().detach().numpy() * len(flows_c)

    rmse = np.sqrt(total_mse / len(test_dataloader.dataset))
    mae = total_mae / len(test_dataloader.dataset)
    mape = total_mape / len(test_dataloader.dataset)

    loss1 = np.sqrt(total_loss1 / len(test_dataloader.dataset))
    space_loss = np.sqrt(total_space_loss / len(test_dataloader.dataset))
    d_space_loss = np.sqrt(total_d_space_loss / len(test_dataloader.dataset))
    time_loss = np.sqrt(total_time_loss / len(test_dataloader.dataset))
    d_time_loss = np.sqrt(total_d_time_loss / len(test_dataloader.dataset))
    sf_loss = np.sqrt(total_sf_loss / len(test_dataloader.dataset))
    poi_loss = total_poi_loss / len(test_dataloader.dataset)
    poif_loss = total_poif_loss / len(test_dataloader.dataset)

    print("RMSE\t{:.2f}\tMAE\t{:.2f}\tMAPE\t{:.4f}".format(rmse, mae, mape))
    print("R_RMSE\t{:.2f}\tPOI_loss\t{:.2f}\tPOIF_loss\t{:.2f}\tS_RMSE\t{:.2f}\tDS_RMSE\t{:.2f}\tT_RMSE\t{:.2f}\tDT_RMSE\t{:.2f}\tSF_RMSE\t{:.2f}".format(loss1, poi_loss, poif_loss, space_loss, d_space_loss, time_loss, d_time_loss, sf_loss))

    f = open('{}/test_result.txt'.format(args.output_dir), 'a')
    f.write("RMSE\t{:.2f}\tMAE\t{:.2f}\tMAPE\t{:.4f}\n".format(rmse, mae, mape))
    f.write(
        "R_RMSE\t{:.2f}\tPOI_loss\t{:.2f}\tPOIF_loss\t{:.2f}\tS_RMSE\t{:.2f}\tDS_RMSE\t{:.2f}\tT_RMSE\t{:.2f}\tDT_RMSE\t{:.2f}\tSF_RMSE\t{:.2f}\n\n".format(
            loss1, poi_loss, poif_loss, space_loss, d_space_loss, time_loss, d_time_loss, sf_loss))
    f.close()

    print_model_parm_nums(c_model, args.c_model, f_model, args.f_model)


def print_model_parm_nums(cmodel, cstr, fmodel, fstr):
    total_cnum = sum([param.nelement() for param in cmodel.parameters()])
    print('{} params: {:.2f} MB'.format(cstr, total_cnum / 1024 / 1024))
    total_fnum = sum([param.nelement() for param in fmodel.parameters()])
    print('{} params: {:.2f} MB'.format(fstr, total_fnum / 1024 / 1024))
    print('total params: {:.2f} MB'.format((total_cnum + total_fnum) / 1024 / 1024))


def Picture_data(id, y, y_hat):
    import numpy as np  # 加载数学库用于函数描述
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import style

    # matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
    plt.figure(figsize=(20, 10), dpi=100)  # 设置图像大小
    font = {'size': 20}

    ax = plt.axes()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())


    X, Y, Y_hat = [], [], []
    n, c, h, w = y.shape
    for i in range(n):
        X.append(i)
        Y.append(y[i][0][4 * 6][4 * 3])
        Y_hat.append(y_hat[i][0][4 * 6][4 * 3])
    ax.plot(X, Y, color='green', lw=2)
    # plt.plot(X, Y_hat, color='red', lw=2, label='inference')
    ax.legend(fontsize=25)
    ax.tick_params(labelsize=25)

    Tmp = ['2016-3-5', '2016-3-7', '2016-3-9', '2016-3-11', '2016-3-13', '2016-3-15']
    # plt.xticks(range(16, 250, 45), Tmp)



    # plt.title('(c) Flow data on TaxiBJ-P4 dataset on region $r_{24, 12} $.', fontsize=34)
    # plt.savefig('Figure/inflow_data_BJ_NYC/TaxiBJ-P4-' + str(id) + '.png', bbox_inches='tight')
    plt.savefig('Figure/inflow_data_BJ_NYC/TaxiBJ-P4-999' + str(id) + '.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.output_dir = args.resume
    args.log_dir = args.output_dir
    args.mask_ratio = 1.0 - 1.0 / args.upscale_factor ** 2
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
