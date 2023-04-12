import torch
import torch.nn as nn
import torch.nn.functional as F
from util import N2_Normalization, Recover_from_density

class sc_pretrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(sc_pretrain, self).__init__()

        # encoder coarse
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.bn_c = nn.BatchNorm2d(base_channels)
        self.conv_c_up = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.linear_c = nn.Sequential(
            nn.Linear(base_channels*4, base_channels*4),
            nn.ReLU(inplace=True)
        )

        # encoder fine
        self.conv_f = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.bn_f = nn.BatchNorm2d(base_channels)
        self.conv_f_up = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.linear_f = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, y):
        enc_c = self.conv_c(x)
        enc_c = self.bn_c(enc_c)
        enc_c = self.conv_c_up(enc_c)
        enc_c = self.linear_c(self.AvgPool(enc_c).squeeze())

        enc_f = self.conv_f(y)
        enc_f = self.bn_f(enc_f)
        enc_f = self.conv_f_up(enc_f)
        enc_f = self.linear_f(self.AvgPool(enc_f).squeeze())

        return enc_c, enc_f

class tc_pretrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(tc_pretrain, self).__init__()

        # encoder time
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.bn_t = nn.BatchNorm2d(base_channels)
        self.conv_t_up = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.linear_t = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        enc_t = self.conv_t(x)
        enc_t = self.bn_t(enc_t)
        enc_t = self.conv_t_up(enc_t)
        enc_t = self.linear_t(self.AvgPool(enc_t).squeeze())

        return enc_t

class fintune(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32, scaler_X=1500, scaler_Y=100):
        super(fintune, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        # encoder coarse
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # encoder time
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # encoder fine-tune
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels*3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Distributional upsampling layers
        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv_c(x)
        x2 = self.conv_t(x)
        x3 = self.conv(x)

        enc_x = torch.cat((x1, x2, x3), dim=1)
        enc_x = self.conv_combin(enc_x)
        out = self.upsampling(enc_x)

        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_nosc(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32, scaler_X=1500, scaler_Y=100):
        super(fintune_nosc, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        # encoder time
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # encoder fine-tune
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Distributional upsampling layers
        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x2 = self.conv_t(x)
        x3 = self.conv(x)

        enc_x = torch.cat((x2, x3), dim=1)
        enc_x = self.conv_combin(enc_x)
        out = self.upsampling(enc_x)

        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_nosctc(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32, scaler_X=1500, scaler_Y=100):
        super(fintune_nosctc, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.img_width = img_width
        self.img_height = img_height

        # encoder fine-tune
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Distributional upsampling layers
        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x3 = self.conv(x)

        enc_x = self.conv_combin(x3)
        out = self.upsampling(enc_x)

        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out