import os

import random
import sys
import warnings
import numpy as np
import argparse
import warnings
from datetime import datetime
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append("..")
from util.metrics import get_MSE, get_MAE, get_MAPE
from util.data_process import get_dataloader, print_model_parm_nums, get_dataloader_inf2

from Models.SCN import SCN
from tensorboardX import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--base_channels', type=int,
                    default=64, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=8,
                    help='image width')
parser.add_argument('--img_height', type=int, default=8,
                    help='image height')
parser.add_argument('--channels', type=int, default=1,
                    help='number of flow image channels')
parser.add_argument('--sample_interval', type=int, default=20,
                    help='interval between validation')
parser.add_argument('--harved_epoch', type=int, default=50,
                    help='halved at every x interval')
parser.add_argument('--upscale_factor', type=int,
                    default=4, help='upscale factor')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--scaler_X', type=int, default=1,
                    help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=1,
                    help='scaler of fine-grained flows')
parser.add_argument('--dataset', type=str, default='TaxiBJ/P1',
                    help='which dataset to use')
parser.add_argument('--fraction', type=int,
                    default=20, help='fraction')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)
random.seed(opt.seed)
warnings.filterwarnings('ignore')
# path for saving model
save_path = 'Saved_model/{}/{}/{}/{}'.format('SCN', opt.dataset, opt.fraction, opt.upscale_factor)
os.makedirs(save_path, exist_ok=True)
# writer = SummaryWriter('runs/{}/{}/{}'.format(opt.dataset, opt.fraction, 'SCN'))

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = SCN(scale=opt.upscale_factor)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)

print_model_parm_nums(model, 'SCN')
criterion = nn.MSELoss()

if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set
datapath = os.path.join('../data', opt.dataset)
train_dataloader, valid_dataloader, test_dataloader = get_dataloader_inf2(
    datapath, opt.scaler_X, opt.scaler_Y, opt.fraction, opt.batch_size, opt.upscale_factor)

# Optimizers
lr = opt.lr
optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))

# training phase
iter = 0
rmses = [np.inf]
maes = [np.inf]
# for epoch in range(opt.n_epochs):
#     train_loss = 0
#     ep_time = datetime.now()
#
#     for i, (flows_c, _, flows_f) in enumerate(train_dataloader):
#         model.train()
#         optimizer.zero_grad()
#
#         # generate images with high resolution
#         gen_hr = model(flows_c)
#         loss = criterion(gen_hr, flows_f)
#
#         loss.backward()
#         optimizer.step()
#
#         print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
#                                                                 opt.n_epochs,
#                                                                 i,
#                                                                 len(train_dataloader),
#                                                                 np.sqrt(loss.item())))
#
#         # counting training mse
#         train_loss += loss.item() * len(flows_c)
#
#         iter += 1
#         # validation phase
#         if iter % opt.sample_interval == 0:
#             model.eval()
#             valid_time = datetime.now()
#             total_mse, total_mae = 0, 0
#             for j, (flows_c, _, flows_f) in enumerate(valid_dataloader):
#                 preds = model(flows_c)
#                 preds = preds.cpu().detach().numpy()
#                 flows_f = flows_f.cpu().detach().numpy()
#                 total_mse += get_MSE(preds, flows_f) * len(flows_c)
#             rmse = np.sqrt(total_mse / len(valid_dataloader.dataset))
#             if rmse < np.min(rmses):
#                 print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now()-valid_time))
#                 torch.save(model.state_dict(),
#                            '{}/final_model.pt'.format(save_path))
#                 f = open('{}/results.txt'.format(save_path), 'a')
#                 f.write("epoch\t{}\titer\t{}\tRMSE\t{:.6f}\n".format(epoch, iter, rmse))
#                 f.close()
#             rmses.append(rmse)
#
#     # halve the learning rate
#     if epoch % opt.harved_epoch == 0 and epoch != 0:
#         lr /= 2
#         optimizer = torch.optim.Adam(
#             model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
#         f = open('{}/results.txt'.format(save_path), 'a')
#         f.write("half the learning rate!\n")
#         f.close()
#
#     print('=================time cost: {}==================='.format(
#         datetime.now()-ep_time))


model.load_state_dict(torch.load('{}/final_model.pt'.format(save_path)))
model.eval()
total_mse, total_mae, total_mape = 0, 0, 0
for j, (flows_c, _, flows_f) in enumerate(test_dataloader):
    preds = model(flows_c)
    preds = preds.cpu().detach().numpy()
    flows_f = flows_f.cpu().detach().numpy()
    total_mse += get_MSE(preds, flows_f) * len(flows_c)
    total_mae += get_MAE(preds, flows_f) * len(flows_c)
    total_mape += get_MAPE(preds, flows_f) * len(flows_c)

    if j == 5:
        plt.figure(figsize=(6.2, 6))
        plt.axis('off')
        ax = sns.heatmap(preds[30][0], cmap="RdYlGn_r", cbar=False)
        plt.savefig('../Figure/SCN.png', bbox_inches='tight')
        plt.show()


rmse = np.sqrt(total_mse / len(test_dataloader.dataset))
mse = total_mse / len(test_dataloader.dataset)
mae = total_mae / len(test_dataloader.dataset)
mape = total_mape / len(test_dataloader.dataset)

with open('{}/test_results.txt'.format(save_path), 'w') as f:
    f.write("RMSE\t{:.4f}\tMAE\t{:.4f}\tMAPE\t{:.4f}\n".format(rmse, mae, mape))
print('Test RMSE = {:.4f}\nMAE = {:.4f}\nMAPE = {:.4f}'.format(rmse, mae, mape))