import torch
import torch.nn as nn
import torch.nn.functional as F
from util import N2_Normalization, Recover_from_density, ResidualBlock

class Strain_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(Strain_net, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
        self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
        self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

        self.ext2lr = nn.Sequential(
            nn.Linear(12, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, img_width * img_height),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels*2, base_channels*2 * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ext):
        ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 5].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(
            ext[:, 6].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, :4]

        ext_out = self.ext2lr(torch.cat(
            [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)

        inp = torch.add(x, ext_out)

        enc_inf = self.conv1(inp)
        enc_pix = self.conv_pix(inp)

        enc_ = self.bn(enc_pix)
        B, C, W, H = enc_.shape
        enc_ = enc_.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
        enc_ = self.linear(enc_)

        enc = torch.cat((enc_inf, enc_pix), dim=1)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out, enc_

class pix_preTrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(pix_preTrain, self).__init__()

        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc = self.conv_pix(x)
        enc = self.bn(enc)

        B, C, W, H = enc.shape
        enc = enc.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
        enc = self.linear(enc)

        return enc


class pix_preTrainE(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, ext_dim=7, img_width=32, img_height=32):
        super(pix_preTrainE, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels+ext_dim, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ext):
        B, C = ext.shape
        ext = ext.view(B, C, 1, 1).contiguous()
        ext = ext.expand(-1, -1, self.img_height, self.img_width)

        x = torch.cat((x, ext), dim=1)

        enc = self.conv_pix(x)
        enc = self.bn(enc)

        B, C, W, H = enc.shape
        enc = enc.permute(0, 2, 3, 1).view(B, -1, C).contiguous()
        enc = self.linear(enc)

        return enc

class tc_preTrain(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(tc_preTrain, self).__init__()

        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = 2.0

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

    def forward(self, x):
        enc = self.conv_tc(x)
        enc = self.bn(enc)

        out = self.linear(self.AvgPool(enc).squeeze())

        return self.normalize(out) * self.alpha

class tc_preTrainE(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, ext_dim=7, img_width=32, img_height=32):
        super(tc_preTrainE, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels+ext_dim, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(base_channels)

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = 2.0

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

    def forward(self, x, ext):
        B, C = ext.shape
        ext = ext.view(B, C, 1, 1).contiguous()
        ext = ext.expand(-1, -1, self.img_height, self.img_width)

        x = torch.cat((x, ext), dim=1)

        enc = self.conv_tc(x)
        enc = self.bn(enc)

        out = self.linear(self.AvgPool(enc).squeeze())

        return self.normalize(out) * self.alpha

class inference_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=8, img_height=8):
        super(inference_net, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.BatchNorm2d(base_channels * 16),
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
        enc = self.conv1(x)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class inference_netE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=8, img_height=8, ext_dim=7):
        super(inference_netE, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels+ext_dim, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.BatchNorm2d(base_channels * 16),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ext):
        B, C = ext.shape
        ext = ext.view(B, C, 1, 1).contiguous()
        ext = ext.expand(-1, -1, self.img_height, self.img_width)

        x_cat = torch.cat((x, ext), dim=1)

        enc = self.conv1(x_cat)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        for i in range(2):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_inf = self.conv1(x)
        enc_tc = self.conv_tc(x)
        enc_pix = self.conv_pix(x)

        enc = torch.cat((enc_inf, enc_tc, enc_pix), dim=1)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_netEmbedding(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32, ext_dim=7):
        super(fintune_netEmbedding, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
        self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
        self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

        self.ext2lr = nn.Sequential(
            nn.Linear(12, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, img_width * img_height),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*3, base_channels*3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels*3, base_channels*3 * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels*3, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ext):
        ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 5].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(
            ext[:, 6].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, :4]

        ext_out = self.ext2lr(torch.cat(
            [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)

        inp = torch.add(x, ext_out)

        enc_inf = self.conv1(inp)
        enc_tc = self.conv_tc(inp)
        enc_pix = self.conv_pix(inp)

        enc = torch.cat((enc_inf, enc_tc, enc_pix), dim=1)
        enc = self.conv2(enc)

        out = self.upsampling(enc)

        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net_inf(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net_inf, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

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
        enc_inf = self.conv1(x)

        enc = enc_inf
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net_pix(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net_pix, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

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
        enc_pix = self.conv_pix(x)

        enc = enc_pix
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net_tc(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net_tc, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

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
        enc_tc = self.conv_tc(x)

        enc = enc_tc
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net_infpix(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net_infpix, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels*2, base_channels*2 * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_inf = self.conv1(x)
        enc_pix = self.conv_pix(x)

        enc = torch.cat((enc_inf, enc_pix), dim=1)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net_tcpix(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net_tcpix, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels*2, base_channels*2 * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_tc = self.conv_tc(x)
        enc_pix = self.conv_pix(x)

        enc = torch.cat((enc_tc, enc_pix), dim=1)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintune_net_inftc(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 img_width=32, img_height=32):
        super(fintune_net_inftc, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        # inference_encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        upsampling = []
        upsampling += [nn.Conv2d(base_channels*2, base_channels*2 * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels*2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc_inf = self.conv1(x)
        enc_tc = self.conv_tc(x)

        enc = torch.cat((enc_inf, enc_tc), dim=1)
        enc = self.conv2(enc)

        out = self.upsampling(enc)
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=1, n_residual_blocks=16, base_channels=64):
        super(Encoder, self).__init__()

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        inp = x

        out1 = self.conv1(inp)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        return out

class BaseModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, img_width=32, img_height=32):
        super(BaseModel, self).__init__()
        self.img_width = img_width
        self.img_height = img_height

        self.encoder = Encoder(in_channels=in_channels, n_residual_blocks=n_residual_blocks,
                               base_channels=base_channels)

        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Distributional upsampling layers
        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

    def forward(self, x):
        out = self.encoder(x)
        out = self.upsampling(out)

        # concatenation backward
        out = self.conv3(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

class fintuneRes_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, img_width=32, img_height=32):
        super(fintuneRes_net, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.scaler_X = 1500
        self.scaler_Y = 100

        self.encoder = Encoder(in_channels=in_channels, n_residual_blocks=n_residual_blocks,
                               base_channels=base_channels)

        # tc_encoder
        self.conv_tc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        # pix_encoder
        self.conv_pix = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # output conv
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Distributional upsampling layers
        upsampling = []
        upsampling += [nn.Conv2d(base_channels, base_channels * 16, 3, 1, 1),
                       nn.PixelShuffle(upscale_factor=4),
                       nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

    def forward(self, x):
        enc_inf = self.encoder(x)
        enc_tc = self.conv_tc(x)
        enc_pix = self.conv_pix(x)
        out = torch.cat((enc_inf, enc_tc, enc_pix), dim=1)
        out = self.conv2(out)

        out = self.upsampling(out)

        # concatenation backward
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)
        return out