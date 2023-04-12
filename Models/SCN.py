from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import functools

import torch
import torch.nn as nn
import torch.nn.init as init
from .utils import N2_Normalization, Recover_from_density


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    total_num = total_num / 1024 / 1024
    print('{} params: {:.3f} MB'.format(str, total_num))

class SCN(nn.Module):

    def __init__(self, scale=4, num_residual_units=128, num_blocks=16, num_channels=1):
        super(SCN, self).__init__()
        self.temporal_size = 1
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 1
        if self.temporal_size:
            num_inputs *= self.temporal_size
        num_outputs = scale * scale * num_channels
        self.num_scales = scale

        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(Head(num_residual_units, self.num_scales))
        for _ in range(num_blocks):
            body.append(
                Block(
                    num_residual_units,
                    kernel_size,
                    width_multiplier=4,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(num_blocks),
                ))
        body.append(Tail(num_residual_units))
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.body = nn.Sequential(*body)

        skip = []
        if num_inputs != num_outputs:
            conv = weight_norm(
                nn.Conv2d(
                    num_inputs,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2))
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

        self.den_softmax = N2_Normalization(scale)
        self.recover = Recover_from_density(scale)

    def forward(self, x):

        input = x

        skip = self.skip(x)
        x_shape = x.shape
        is_padding = False
        if x.shape[-1] % (2 ** self.num_scales) or x.shape[-2] % (2 ** self.num_scales):
            pad_h = (-x.shape[-2]) % (2 ** self.num_scales)
            pad_w = (-x.shape[-1]) % (2 ** self.num_scales)
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), 'replicate')
            is_padding = True
        x = self.body(x)
        if is_padding:
            x = x[..., :x_shape[-2], :x_shape[-1]]
        x = x + skip
        x = self.shuf(x)

        out = self.den_softmax(x)
        out = self.recover(out, input)


        return out


class Head(nn.Module):

    def __init__(self, n_feats, num_scales):
        super(Head, self).__init__()
        self.num_scales = num_scales

        down = []
        down.append(nn.UpsamplingBilinear2d(scale_factor=0.5))
        self.down = nn.Sequential(*down)

    def forward(self, x):
        x_list = [x]
        for _ in range(self.num_scales - 1):
            x_list.append(self.down(x_list[-1]))
        return x_list


class Block(nn.Module):

    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 width_multiplier=1,
                 weight_norm=torch.nn.utils.weight_norm,
                 res_scale=1):
        super(Block, self).__init__()
        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(True))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

        down = []
        down.append(
            weight_norm(nn.Conv2d(num_residual_units, num_residual_units, 1)))
        down.append(nn.UpsamplingBilinear2d(scale_factor=0.5))
        self.down = nn.Sequential(*down)

        up = []
        up.append(weight_norm(nn.Conv2d(num_residual_units, num_residual_units, 1)))
        up.append(nn.UpsamplingBilinear2d(scale_factor=2.0))
        self.up = nn.Sequential(*up)

    def forward(self, x_list):
        res_list = [self.body(x) for x in x_list]
        down_res_list = [res_list[0]] + [self.down(x) for x in res_list[:-1]]
        up_res_list = [self.up(x) for x in res_list[1:]] + [res_list[-1]]
        x_list = [
            x + r + d + u
            for x, r, d, u in zip(x_list, res_list, down_res_list, up_res_list)
        ]
        return x_list


class Tail(nn.Module):

    def __init__(self, n_feats):
        super(Tail, self).__init__()

    def forward(self, x_list):
        return x_list[0]

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    print(input.shape)
    model = SCN()
    output = model(input)
    print(output.shape)
    print_model_parm_nums(model, 'SCN')