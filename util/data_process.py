import random
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import math
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

sys.path.append(".")
from util.metrics import get_MAPE, get_MAE, get_MSE

seed = 2022
torch.manual_seed(seed)
random.seed(seed)


def get_dataloader(datapath, scaler_X, scaler_Y, fraction, batch_size, mode='train'):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X = np.load(os.path.join(datapath, 'X.npy'))
    Y = np.load(os.path.join(datapath, 'Y.npy'))
    ext = np.load(os.path.join(datapath, 'ext.npy'))

    if mode == 'train' and fraction != 0:
        length = len(X)
        sample_index = int(length * fraction / 100)
        X = X[:sample_index]
        Y = Y[:sample_index]
        ext = ext[:sample_index]

    X = Tensor(np.expand_dims(X, 1)) / scaler_X
    Y = Tensor(np.expand_dims(Y, 1)) / scaler_Y
    ext = Tensor(ext)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def get_dataloader_inf(datapath, scaler_X, scaler_Y, fraction, batch_size, mode='train'):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X = np.load(os.path.join(datapath, '8X.npy'))
    Y = np.load(os.path.join(datapath, 'X.npy'))
    ext = np.load(os.path.join(datapath, 'ext.npy'))

    if fraction != 0:
        _, h, w = X.shape
        sample_index_len = int(h * w * fraction / 100)
        sample_index = random.sample(range(0, h * w), sample_index_len)
        for i in range(sample_index_len):
            p = int(sample_index[i] / w)
            q = int(sample_index[i] % w)
            X[:][p][q] = -1
            print(p, ', ', q)

    X = Tensor(np.expand_dims(X, 1)) / scaler_X
    Y = Tensor(np.expand_dims(Y, 1)) / scaler_Y
    ext = Tensor(ext)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def get_dataloader_inf2(datapath, scaler_X, scaler_Y, fraction, batch_size, upscale_factor):
    train_datapath = os.path.join(datapath, 'train')
    valid_datapath = os.path.join(datapath, 'valid')
    test_datapath = os.path.join(datapath, 'test')

    if upscale_factor == 4:
        tY = 'X.npy'
    else:
        tY = '16X.npy'

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    train_X = np.load(os.path.join(train_datapath, '8X.npy'))
    train_Y = np.load(os.path.join(train_datapath, tY))
    train_ext = np.load(os.path.join(train_datapath, '8X.npy'))
    valid_X = np.load(os.path.join(valid_datapath, '8X.npy'))
    valid_Y = np.load(os.path.join(valid_datapath, tY))
    valid_ext = np.load(os.path.join(valid_datapath, '8X.npy'))
    test_X = np.load(os.path.join(test_datapath, '8X.npy'))
    test_Y = np.load(os.path.join(test_datapath, tY))
    test_ext = np.load(os.path.join(test_datapath, '8X.npy'))

    if fraction != 0:
        _, h, w = train_X.shape
        sample_index_len = int(h * w * fraction / 100)
        flag = [[0 for i in range(w)] for j in range(h)]
        candidate = []
        for i in range(h * w):
            p = int(i / w)
            q = int(i % w)
            # print(p,",  " , q)
            Flag = 0
            for m in range(-1, 2):
                if (p + m >= 0 and p + m < h and flag[(p + m)][q] == 1) or (
                        q + m >= 0 and q + m < w and flag[p][(q + m)] == 1):
                    Flag = 1
                    break
            if Flag == 1:
                continue
            else:
                flag[p][q] = 1
                candidate.append(i)

        # t = [[0 for i in range(w)] for j in range(h)]
        sample_len = sample_index_len
        if sample_index_len > len(candidate):
            sample_len = len(candidate)
        sample_index = random.sample(candidate, sample_len)
        print(sample_index)
        print(len(sample_index))
        for i in range(len(sample_index)):
            p = int(sample_index[i] / w)
            q = int(sample_index[i] % w)
            for j in range(train_X.shape[0]):
                train_X[j][p][q] = -1
            for j in range(valid_X.shape[0]):
                valid_X[j][p][q] = -1
            for j in range(test_X.shape[0]):
                test_X[j][p][q] = -1
            # t[p][q] = 1

        Mean(train_X)
        Mean(valid_X)
        Mean(test_X)

        # complete_Mean_Tetst(sample_index, test_X, test_Y)

    # sys.exit(-1)
    # return train_X, train_Y, test_X, test_Y


    train_X = Tensor(np.expand_dims(train_X, 1)) / scaler_X
    train_Y = Tensor(np.expand_dims(train_Y, 1)) / scaler_Y
    train_ext = Tensor(train_ext)
    valid_X = Tensor(np.expand_dims(valid_X, 1)) / scaler_X
    valid_Y = Tensor(np.expand_dims(valid_Y, 1)) / scaler_Y
    valid_ext = Tensor(valid_ext)
    test_X = Tensor(np.expand_dims(test_X, 1)) / scaler_X
    test_Y = Tensor(np.expand_dims(test_Y, 1)) / scaler_Y
    test_ext = Tensor(test_ext)

    train_data = torch.utils.data.TensorDataset(train_X, train_ext, train_Y)
    valid_data = torch.utils.data.TensorDataset(valid_X, valid_ext, valid_Y)
    test_data = torch.utils.data.TensorDataset(test_X, test_ext, test_Y)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

# 数据补全的数据加载函数
def get_dataloader_inf3(datapath, scaler_X, scaler_Y, fraction, batch_size, len_closeness, len_period, len_trend, T, upscale_factor):
    train_datapath = os.path.join(datapath, 'train')
    valid_datapath = os.path.join(datapath, 'valid')
    test_datapath = os.path.join(datapath, 'test')

    if upscale_factor == 4:
        zz = 'X.npy'
    else:
        zz = '16X.npy'

    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    train_X = np.load(os.path.join(train_datapath, '8X.npy'))
    train_Y = np.load(os.path.join(train_datapath, '8X.npy'))
    train_Z = np.load(os.path.join(train_datapath, zz))
    valid_X = np.load(os.path.join(valid_datapath, '8X.npy'))
    valid_Y = np.load(os.path.join(valid_datapath, '8X.npy'))
    valid_Z = np.load(os.path.join(valid_datapath, zz))
    test_X = np.load(os.path.join(test_datapath, '8X.npy'))
    test_Y = np.load(os.path.join(test_datapath, '8X.npy'))
    test_Z = np.load(os.path.join(test_datapath, zz))

    # Picture_data(test_X)

    n, h, w = train_X.shape

    # 归一化到 [0，1]
    mmn = MinMaxNormalization()
    # train_X = train_X.reshape(n * h * w)
    # mmn.fit(train_X)
    # train_X = train_X.reshape(n, h, w)
    #
    # for i in range(train_X.shape[0]):
    #     for j in range(train_X.shape[1]):
    #         for k in range(train_X.shape[2]):
    #             train_X[i][j][k] = mmn.transform(train_X[i][j][k])
    #             train_Y[i][j][k] = mmn.transform(train_Y[i][j][k])
    #
    #             if i < valid_X.shape[0]:
    #                 valid_X[i][j][k] = mmn.transform(valid_X[i][j][k])
    #                 valid_Y[i][j][k] = mmn.transform(valid_Y[i][j][k])
    #             if i < test_X.shape[0]:
    #                 test_X[i][j][k] = mmn.transform(test_X[i][j][k])
    #                 test_Y[i][j][k] = mmn.transform(test_Y[i][j][k])
    #
    # for i in range(train_Z.shape[0]):
    #     for j in range(train_Z.shape[1]):
    #         for k in range(train_Z.shape[2]):
    #             train_Z[i][j][k] = mmn.transform(train_Z[i][j][k])
    #             if i < valid_Z.shape[0]:
    #                 valid_Z[i][j][k] = mmn.transform(valid_Z[i][j][k])
    #             if i < test_Z.shape[0]:
    #                 test_Z[i][j][k] = mmn.transform(test_Z[i][j][k])


    sample_index = []
    # miss data=0
    if fraction != 0:
        sample_index_len = int(h * w * fraction / 100)
        flag = [[0 for i in range(w)] for j in range(h)]
        candidate = []
        for i in range(h * w):
            p = int(i / w)
            q = int(i % w)
            # print(p,",  " , q)
            Flag = 0
            for m in range(-1, 2):
                if (p + m >= 0 and p + m < h and flag[(p + m)][q] == 1) or (
                        q + m >= 0 and q + m < w and flag[p][(q + m)] == 1):
                    Flag = 1
                    break
            if Flag == 1:
                continue
            else:
                flag[p][q] = 1
                candidate.append(i)

        # t = [[0 for i in range(w)] for j in range(h)]
        sample_len = sample_index_len
        if sample_index_len > len(candidate):
            sample_len = len(candidate)
        sample_index = random.sample(candidate, sample_len)
        for i in range(len(sample_index)):
            p = int(sample_index[i] / w)
            q = int(sample_index[i] % w)
            for j in range(train_X.shape[0]):
                train_X[j][p][q] = -1
            for j in range(valid_X.shape[0]):
                valid_X[j][p][q] = -1
            for j in range(test_X.shape[0]):
                test_X[j][p][q] = -1
            # t[p][q] = 1

        # table = [str(',\t'.join(str(row).split(','))) for row in t]
        # print('\n'.join(table))

        # test_X = test_X.astype(int)
        # test_Y = test_Y.astype(int)
        # print(test_X[0])
        # print("\n\n")
        # print(test_Y[0])
        # sys.exit(0)

        # Mean(train_X)
        # Mean(valid_X)
        # Mean(test_X)

    train_X = np.expand_dims(train_X, 1)
    valid_X = np.expand_dims(valid_X, 1)
    test_X = np.expand_dims(test_X, 1)
    train_Y = np.expand_dims(train_Y, 1)
    train_Z = np.expand_dims(train_Z, 1)
    valid_Y = np.expand_dims(valid_Y, 1)
    valid_Z = np.expand_dims(valid_Z, 1)
    test_Y = np.expand_dims(test_Y, 1)
    test_Z = np.expand_dims(test_Z, 1)


    train_HX, train_X, train_Y, train_Z = history_data(train_X, train_Y, train_Z, len_closeness, len_period, len_trend, T)
    valid_HX, valid_X, valid_Y, valid_Z = history_data(valid_X, valid_Y, valid_Z, len_closeness, len_period, len_trend, T)
    test_HX, test_X, test_Y, test_Z = history_data(test_X, test_Y, test_Z, len_closeness, len_period, len_trend, T)

    if len(train_HX) == 1:
        train_data = torch.utils.data.TensorDataset(torch.Tensor(train_HX[0]),
                                                    train_X, train_Y, train_Z)
        valid_data = torch.utils.data.TensorDataset(torch.Tensor(valid_HX[0]),
                                                    valid_X, valid_Y, valid_Z)
        test_data = torch.utils.data.TensorDataset(torch.Tensor(test_HX[0]),
                                                   test_X, test_Y, test_Z)
    elif len(train_HX) == 2:
        train_data = torch.utils.data.TensorDataset(torch.Tensor(train_HX[0]), torch.Tensor(train_HX[1]),
                                                    train_X, train_Y, train_Z)
        valid_data = torch.utils.data.TensorDataset(torch.Tensor(valid_HX[0]), torch.Tensor(valid_HX[1]),
                                                    valid_X, valid_Y, valid_Z)
        test_data = torch.utils.data.TensorDataset(torch.Tensor(test_HX[0]), torch.Tensor(test_HX[1]),
                                                   test_X, test_Y, test_Z)

    else:
        train_data = torch.utils.data.TensorDataset(torch.Tensor(train_HX[0]), torch.Tensor(train_HX[1]),
                                                    torch.Tensor(train_HX[2]), train_X, train_Y, train_Z)
        valid_data = torch.utils.data.TensorDataset(torch.Tensor(valid_HX[0]), torch.Tensor(valid_HX[1]),
                                                    torch.Tensor(valid_HX[2]), valid_X, valid_Y, valid_Z)
        test_data = torch.utils.data.TensorDataset(torch.Tensor(test_HX[0]), torch.Tensor(test_HX[1]),
                                                   torch.Tensor(test_HX[2]), test_X, test_Y, test_Z)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    return train_dataloader, valid_dataloader, test_dataloader, sample_index, mmn


def history_data(train_X, train_Y, train_Z, len_closeness, len_period, len_trend, T):
    XC, XP, XT, Y = [], [], [], []
    XY, XZ = [], []
    for i in range(len(train_X)):
        index_c = [index for index in range(i - len_closeness, i)]
        index_p = [index for index in range(i - len_period * T, i, T)]
        index_t = [index for index in range(i - len_trend * T * 7, i, T * 7)]

        if ((len(index_t) > 0 and min(index_t) >= 0) or index_t == []) and i % T >= len_closeness:
            xc = [train_X[index] for index in index_c]
            if len(xc) != 0:
                xc = np.stack(xc, axis=0)
                XC.append(xc)

            xp = [train_X[index] for index in index_p]
            if len(xp) != 0:
                xp = np.stack(xp, axis=0)
                XP.append(xp)

            xt = [train_X[index] for index in index_t]
            if len(xt) != 0:
                xt = np.stack(xt, axis=0)
                XT.append(xt)

            if len(xc) + len(xp) + len(xt) != 0:
                Y.append(train_X[i])
                XY.append(train_Y[i])
                XZ.append(train_Z[i])

    if len(XC) != 0:
        XC = np.stack(XC, axis=0)
    if len(XP) != 0:
        XP = np.stack(XP, axis=0)
    if len(XT) != 0:
        XT = np.stack(XT, axis=0)

    Y = np.stack(Y, axis=0)
    XY = np.stack(XY, axis=0)
    XZ = np.stack(XZ, axis=0)

    HX = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC, XP, XT]):
        if l > 0:
            HX.append(X_)

    return HX, torch.Tensor(Y), torch.Tensor(XY), torch.Tensor(XZ)


class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * X / self._max
        # X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * self._max
        # X = 1. * X * (self._max - self._min) + self._min
        return X
    def unfit(self):
        return self._max


def Mean(data):
    n, h, w = data.shape
    for i in range(h):
        for j in range(w):
            if data[0][i][j] != -1:
                continue

            for k in range(n):
                num = 0
                nn = 0
                for p in range(-1, 2):
                    for q in range(-1, 2):
                        if i + p < 0 or i + p >= h or j + q < 0 or j + q >= w or data[k][(i + p)][(j + q)] == -1:
                            continue
                        num = num + data[k][(i + p)][(j + q)]
                        nn += 1
                if nn == 0:
                    continue
                else:
                    data[k][i][j] = num / nn


def Picture_data(data):
    import numpy as np  # 加载数学库用于函数描述
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import style

    # matplotlib.rcParams['text.usetex'] = True  # 开启Latex风格
    plt.figure(figsize=(20, 10), dpi=70)  # 设置图像大小
    X, Y = [], []
    n, h, w = data.shape
    for i in range(200):
        X.append(i)
        Y.append(data[i][4][4])
    plt.plot(X, Y, '-p', color='grey',
        marker = 'o',
        markersize=8, linewidth=3,
        markerfacecolor='red',
        markeredgecolor='grey',
        markeredgewidth=3)
    plt.show()

def get_dataloader_apn(datapath, batch_size, fraction, mode='train'):
    datapath = os.path.join(datapath, mode)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    anchor = np.load(os.path.join(datapath, 'anchor.npy'))
    pos = np.load(os.path.join(datapath, 'pos.npy'))
    neg = np.load(os.path.join(datapath, 'neg.npy'))

    if mode == 'train' and fraction != 100:
        length = len(anchor)
        sample_index = int(length * fraction / 100)
        anchor = anchor[:sample_index]
        pos = pos[:sample_index]
        neg = neg[:sample_index]

    anchor = Tensor(np.expand_dims(anchor, 1))
    pos = Tensor(np.expand_dims(pos, 1))
    neg = Tensor(np.expand_dims(neg, 1))

    assert len(anchor) == len(pos)
    print('# {} samples: {}'.format(mode, len(anchor)))

    data = torch.utils.data.TensorDataset(anchor, pos, neg)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))

def create_tc_data_HardSample(type, mode):
    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))

    anchor = []
    p1 = []
    p2 = []

    length = len(A)
    for i in range(length):
        maxn = -float('inf')
        minn = float('inf')
        kmax, kmin = i, i
        for j in range(length):
            if i == j:
                continue
            dist = np.sqrt(np.sum(np.square(A[i] - A[j])))

            if dist > maxn:
                maxn = dist
                kmax = j

            if dist < minn:
                minn = dist
                kmin = j

        # print(minn, maxn, i, kmin, kmax)

        anchor.append(A[i])
        p1.append(A[kmin])
        p2.append(A[kmax])

    anchor = np.array(anchor)
    p1 = np.array(p1)
    p2 = np.array(p2)

    anchor_path = os.path.join(datapath, 'anchor.npy')
    pos_path = os.path.join(datapath, 'pos.npy')
    neg_path = os.path.join(datapath, 'neg.npy')

    np.save(anchor_path, anchor)
    np.save(pos_path, p1)
    np.save(neg_path, p2)

def create_tc_data_WeightSample(type, mode, k=5):
    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'X.npy'))

    anchor = []
    p1 = []
    p2 = []

    length = len(A)
    for i in range(length):
        dis_dict = {}

        for j in range(length):
            if i == j:
                continue
            dist = np.sqrt(np.sum(np.square(A[i] - A[j])))
            dis_dict[j] = dist

        dis_dict = sorted(dis_dict.items(), key=lambda item:item[1])
        dis_pos = dis_dict[:k]
        dis_neg = dis_dict[-k:]

        pos_sum, neg_sum = 0, 0

        for j in dis_pos:
            pos_sum += 1./j[1]

        for j in dis_neg:
            neg_sum += j[1]

        pos_zero, neg_zero = 0, 0
        for j in dis_pos:
            pos_zero += (1./j[1])/pos_sum * A[j[0]]
        for j in dis_neg:
            neg_zero += j[1]/neg_sum * A[j[0]]

        anchor.append(A[i])
        p1.append(pos_zero)
        p2.append(neg_zero)

    anchor = np.array(anchor)
    p1 = np.array(p1)
    p2 = np.array(p2)

    anchor_path = os.path.join(datapath, str(k)+'anchor.npy')
    pos_path = os.path.join(datapath, str(k)+'pos.npy')
    neg_path = os.path.join(datapath, str(k)+'neg.npy')

    np.save(anchor_path, anchor)
    np.save(pos_path, p1)
    np.save(neg_path, p2)

def create_scaler_data(type, mode, up_size):

    datapath = os.path.join('../data', mode)
    datapath = os.path.join(datapath, type)
    A = np.load(os.path.join(datapath, 'Y.npy'))[:, :32, :32]
    (x, y, z) = A.shape
    print(A.shape)
    np.save(os.path.join(datapath, 'X.npy'), A)

    zeros = np.zeros(shape=(x, y//up_size, z//up_size))
    for k in range(x):
        for i in range(y):
            for j in range(z):
                ii = i//up_size
                jj = j//up_size
                zeros[k][ii][jj] += A[k][i][j]

    if mode == 'BikeNYC':
        if up_size == 2:
            bikename = '16X.npy'
        else:
            bikename = '8X.npy'
        temp_path = os.path.join(datapath, bikename)
        np.save(temp_path, zeros)
        print(zeros.shape)
    else:
        temp_path = os.path.join(datapath, '16X.npy')
        np.save(temp_path, zeros)


def complete_Mean_Tetst(sample_index, test_pred, test_true):
    n, h, w = test_pred.shape
    mask_preds = np.zeros((n, len(sample_index)))
    mask_flows_y = np.zeros((n, len(sample_index)))
    for i in range(len(sample_index)):
        p = int(sample_index[i] / w)
        q = int(sample_index[i] % w)
        mask_preds[:, i] = test_pred[:, p, q]
        mask_flows_y[:, i] = test_true[:, p, q]

    mse = get_MSE(mask_preds, mask_flows_y)
    mae = get_MAE(mask_preds, mask_flows_y)
    mape = get_MAPE(mask_preds, mask_flows_y)
    rmse = np.sqrt(mse)

    print('Test RMSE = {:.6f}, MSE = {:.6f}, MAE = {:.6f}, MAPE = {:.6f}'.format(rmse, mse, mae, mape))

if __name__ == '__main__':
    # for i in range(1,3):
    #     create_scaler_data('train', 'BikeNYC', i * 2)


    # train_dataloader, valid_dataloader, test_dataloader, _, mmn = get_dataloader_inf3('../data/TaxiBJ/P1', 1, 1, 20, 16, 2, 1, 1, 28)
    # print(len(train_dataloader.dataset))
    # print(len(valid_dataloader.dataset))
    # print(len(test_dataloader.dataset))

    get_dataloader_inf2('../data/TaxiBJ/P1', 1, 1, 60, 16, 4)

    # datapath = os.path.join('../data', 'POI_32.npy')
    # A = np.load(datapath)
    # (x, y, z) = A.shape
    # print(A.shape)
    #
    # zeros = np.zeros(shape=(x//2, y//2, z))
    # for k in range(x):
    #     for i in range(y):
    #         for j in range(z):
    #             ii = i//2
    #             kk = k//2
    #             zeros[kk][ii][j] += A[k][i][j]
    # temp_path = os.path.join('../data', 'POI_16.npy')
    # np.save(temp_path, zeros)
    # print(zeros.shape)