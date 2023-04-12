import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import N2_Normalization, Recover_from_density, ResidualBlock
import math

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Pix_preTrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(Pix_preTrain, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
        )

        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc = self.conv(x)

        B, C, W, H = enc.shape
        enc = enc.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
        enc = self.linear(enc)
        return enc

class TC_preTrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(TC_preTrain, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
        )

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)

        enc_tensor = self.linear(self.AvgPool(out).squeeze())

        return enc_tensor

class Encoder(nn.Module):
    def __init__(self, in_channels=1, n_residual_blocks=16, base_channels=64):
        super(Encoder, self).__init__()

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.BatchNorm2d(base_channels))

    def forward(self, x):
        inp = x

        out1 = self.conv1(inp)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        return out

class UrbanFM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, img_width=32, img_height=32, scaler_X=1500, scaler_Y=100, upscale_factor=2):
        super(UrbanFM, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height
        self.upscale_factor = upscale_factor

        self.encoder = Encoder(in_channels=in_channels, n_residual_blocks=n_residual_blocks,
                               base_channels=base_channels)

        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(int(self.upscale_factor/2)):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(self.upscale_factor)
        self.recover = Recover_from_density(self.upscale_factor)

    def forward(self, x):
        out = self.encoder(x)
        out = self.upsampling(out)

        # concatenation backward
        out = self.conv3(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)
        return out

class UrbanFM_fintune(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, img_width=32, img_height=32, scaler_X=1500, scaler_Y=100):
        super(UrbanFM_fintune, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        self.encoder = Encoder(in_channels=in_channels, n_residual_blocks=n_residual_blocks,
                               base_channels=base_channels)

        self.conv_P = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

    def forward(self, x):
        out = self.encoder(x)
        out_P = self.conv_P(x)

        out = torch.cat((out, out_P), dim=1)
        out = self.conv_combin(out)

        out = self.upsampling(out)

        # concatenation backward
        out = self.conv3(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)
        return out

class UrbanFM_fintune_all(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, img_width=32, img_height=32, scaler_X=1500, scaler_Y=100):
        super(UrbanFM_fintune_all, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        self.encoder = Encoder(in_channels=in_channels, n_residual_blocks=n_residual_blocks,
                               base_channels=base_channels)

        self.conv_P = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
        )

        self.conv_T = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

    def forward(self, x):
        out = self.encoder(x)
        out_P = self.conv_P(x)
        out_T = self.conv_T(x)

        out = torch.cat((out, out_P, out_T), dim=1)
        out = self.conv_combin(out)

        out = self.upsampling(out)

        # concatenation backward
        out = self.conv3(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)
        return out