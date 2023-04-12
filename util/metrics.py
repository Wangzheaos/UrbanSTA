import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import sys

seed = 2022
torch.manual_seed(seed)
random.seed(seed)


def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_RMSE(pred, real):
    return math.sqrt(np.mean(np.power(real - pred, 2)))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))
    
def get_MAPE(pred, real, upscale_factor=4):
    # pred += 1
    # real += 1
    ori_real = real.copy()
    epsilon = 1 # if use small number like 1e-5 resulting in very large value
    real[real == 0] = epsilon
    return np.mean(np.abs(ori_real - pred) / real)

def distance_tensor(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)


def loss_c(C_enc, poi, margin, Type):
    B, WH, C = C_enc.shape
    length = WH

    Loss = torch.zeros(1).cuda()
    B_logits, B_labels = [], []
    for K in range(B):
        num = length
        random_index = random.sample(range(0, length), num)

        X = C_enc[K, random_index[0], :]
        Y = C_enc[K, random_index, :]
        logits = torch.matmul(X.unsqueeze(0), torch.transpose(Y, 0, 1)).squeeze()
        logits = torch.where(torch.isinf(logits), torch.full_like(logits, 0), logits)

        # X = X.repeat(num, 1)
        # diff = torch.abs(X - Y)
        # ans = torch.pow(diff, 2).mean(-1)

        poi = poi.reshape(WH)
        poi_x = poi[random_index[0]].repeat(WH)
        poi_y = poi[random_index]

        ans = torch.abs(poi_x - poi_y)
        # ans = torch.pow(diff, 2)

        # print(torch.sum(ans))
        ans = (ans < margin).long().cuda()

        # print(torch.sum(ans))
        B_logits.append(logits)
        B_labels.append(ans)


    B_logits = torch.stack(B_logits, dim=0)
    B_labels = torch.stack(B_labels, dim=0)

    if Type == 'sigmoid':
        crition = torch.nn.MultiLabelSoftMarginLoss()
        loss = crition(B_logits, B_labels)
        Loss += loss

    elif Type == 'softmax':
        B_logits = (1 - 2 * B_labels) * B_logits
        logits_neg = B_logits - B_labels * 1e12
        logits_pos = B_logits - (1 - B_labels) * 1e12
        zeros = torch.zeros_like(B_logits[..., :1])

        logits_neg = torch.cat((logits_neg, zeros), dim=-1)
        logits_pos = torch.cat((logits_pos, zeros), dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = (neg_loss + pos_loss).mean()
        Loss += loss

    return Loss[0]


def loss_fine_poi(C_enc, poi, margin, Type):
    B, WH, C = C_enc.shape
    length = WH

    Loss = torch.zeros(1).cuda()
    B_logits, B_labels = [], []
    for K in range(B):
        num = length
        random_index = random.sample(range(0, length), num)

        for n in range(10):
            X = C_enc[K, random_index[n], :]
            Y = C_enc[K, random_index, :]

            logits = torch.matmul(X.unsqueeze(0), torch.transpose(Y, 0, 1)).squeeze()
            logits = torch.where(torch.isinf(logits), torch.full_like(logits, 0), logits)


            # X = X.repeat(num, 1)
            # diff = torch.abs(X - Y)
            # ans = torch.pow(diff, 2).mean(-1)

            poi = poi.reshape(WH)
            poi_x = poi[random_index[n]].repeat(WH)
            poi_y = poi[random_index]

            ans = torch.abs(poi_x - poi_y)
            # ans = torch.pow(diff, 2)

            # print(torch.sum(ans))
            ans = (ans < margin).long().cuda()
            # if margin != 100:
            #     print(torch.sum(ans))
            B_logits.append(logits)
            B_labels.append(ans)

    B_logits = torch.stack(B_logits, dim=0)
    B_labels = torch.stack(B_labels, dim=0)

    if Type == 'sigmoid':
        crition = torch.nn.MultiLabelSoftMarginLoss()
        loss = crition(B_logits, B_labels)
        Loss += loss

    elif Type == 'softmax':
        B_logits = (1 - 2 * B_labels) * B_logits
        logits_neg = B_logits - B_labels * 1e12
        logits_pos = B_logits - (1 - B_labels) * 1e12
        zeros = torch.zeros_like(B_logits[..., :1])

        logits_neg = torch.cat((logits_neg, zeros), dim=-1)
        logits_pos = torch.cat((logits_pos, zeros), dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = (neg_loss + pos_loss).mean()
        Loss += loss

    return Loss[0]