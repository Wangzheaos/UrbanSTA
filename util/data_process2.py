import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import math
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from metrics import get_MAE, get_MSE, get_MAPE
from data_process import get_dataloader_inf2

def HA(fraction=100, mode='P1'):
    up_size = 4
    name = '8X.npy'
    if mode == 'BikeNYC':
        up_size = 2
        name = '10X.npy'
    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, 'train')
    A = np.load(os.path.join(datapath, name))
    B = np.load(os.path.join(datapath, 'X.npy'))
    print("train A:", A.shape)
    print("train B:", B.shape)

    if fraction != 100:
        length = len(A)
        sample_index = int(length * fraction / 100)
        # A = A[sample_index*1:sample_index*2]
        # B = B[sample_index*1:sample_index*2]

        A = A[-sample_index:]
        B = B[-sample_index:]

    (x, y, z) = B.shape
    Aa = np.zeros(shape=(x, y, z))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = int(i / up_size)
                jj = int(j / up_size)
                if A[k][ii][jj] == 0:
                    continue
                Aa[k][i][j] = B[k][i][j] / A[k][ii][jj]
    size = np.zeros(shape=(y, z))
    for i in range(y):
        for j in range(z):
            pp = 0
            for k in range(x):
                pp += Aa[k][i][j]
            size[i][j] = pp / x

    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, 'test')
    A = np.load(os.path.join(datapath, name))
    B = np.load(os.path.join(datapath, 'X.npy'))
    print("test A:", A.shape)
    print("test B:", B.shape)

    (x, y, z) = B.shape
    C = np.zeros(shape=(x, y, z))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = int(i / up_size)
                jj = int(j / up_size)
                if A[k][ii][jj] == 0:
                    continue
                C[k][i][j] = A[k][ii][jj]

    preds = np.zeros(shape=(x, y, z))
    for k in range(x):
        preds[k] = size * C[k]

    mse = get_MSE(preds, B)
    mae = get_MAE(preds, B)
    mape = get_MAPE(preds, B)
    rmse = np.sqrt(mse)

    print('Test RMSE = {:.3f}, MAE = {:.3f}, MAPE = {:.3f}'.format(rmse, mae, mape))

def cal_Mean(mode='P1'):
    up_size = 4
    name = '8X.npy'
    if mode == 'BikeNYC':
        up_size = 2
        name = '10X.npy'
    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, 'test')
    A = np.load(os.path.join(datapath, name))
    B = np.load(os.path.join(datapath, 'X.npy'))
    (x, y, z) = B.shape
    print(A.shape)
    print(B.shape)
    preds = np.zeros(shape=(x,y,z))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = int(i/up_size)
                jj = int(j/up_size)
                preds[k][i][j] = A[k][ii][jj]/(up_size ** 2)

    mse = get_MSE(preds, B)
    mae = get_MAE(preds, B)
    mape = get_MAPE(preds, B)
    rmse = np.sqrt(mse)

    print('Test RMSE = {:.6f}, MSE = {:.6f}, MAE = {:.6f}, MAPE = {:.6f}'.format(rmse, mse, mae, mape))


def cal_Mean_uc_f(fraction=20, mode='data/TaxiBJ/P2', up_size=2):
    print(mode, '   ', up_size, '  ', fraction)

    _, _, A, B = get_dataloader_inf2(mode, 1, 1, fraction, 8, up_size)
    (x, y, z) = B.shape
    print(A.shape)
    print(B.shape)
    preds = np.zeros(shape=(x,y,z))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = int(i/up_size)
                jj = int(j/up_size)
                preds[k][i][j] = A[k][ii][jj]/(up_size ** 2)

    mse = get_MSE(preds, B)
    mae = get_MAE(preds, B)
    mape = get_MAPE(preds, B)
    rmse = np.sqrt(mse)

    print('Test RMSE = {:.2f}, MSE = {:.2f}, MAE = {:.2f}, MAPE = {:.4f}'.format(rmse, mse, mae, mape))


def HA_uc_f(fraction=20, mode='data/TaxiBJ/P3', up_size=2):

    A, B, C, D = get_dataloader_inf2(mode, 1, 1, fraction, 8, up_size)
    print(mode)

    print("train A:", A.shape)
    print("train B:", B.shape)

    (x, y, z) = B.shape
    Aa = np.zeros(shape=(x, y, z))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = int(i / up_size)
                jj = int(j / up_size)
                if A[k][ii][jj] == 0:
                    continue
                Aa[k][i][j] = B[k][i][j] / A[k][ii][jj]
    size = np.zeros(shape=(y, z))
    for i in range(y):
        for j in range(z):
            pp = 0
            for k in range(x):
                pp += Aa[k][i][j]
            size[i][j] = pp / x


    A = C
    B = D
    print("test A:", A.shape)
    print("test B:", B.shape)

    (x, y, z) = B.shape
    C = np.zeros(shape=(x, y, z))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = int(i / up_size)
                jj = int(j / up_size)
                if A[k][ii][jj] == 0:
                    continue
                C[k][i][j] = A[k][ii][jj]

    preds = np.zeros(shape=(x, y, z))
    for k in range(x):
        preds[k] = size * C[k]

    mse = get_MSE(preds, B)
    mae = get_MAE(preds, B)
    mape = get_MAPE(preds, B)
    rmse = np.sqrt(mse)

    print('Test RMSE = {:.2f}, MAE = {:.2f}, MAPE = {:.4f}'.format(rmse, mae, mape))


if __name__ == '__main__':
    # HA_uc_f(fraction=60, mode='data/BikeNYC', up_size=4)
    cal_Mean_uc_f(fraction=60, mode='data/BikeNYC', up_size=4)