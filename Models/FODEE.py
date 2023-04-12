import torch
import torch.nn as nn
import torch.nn.functional as F
from util import N2_Normalization, Recover_from_density

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class FODEE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, n_residual_blocks=16,
                 img_width=32, img_height=32):
        super(FODEE, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
        self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
        self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

        self.ext2lr = nn.Sequential(
            nn.Linear(12, 128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, img_width * img_height),
            nn.ReLU(inplace=True)
        )

        self.ext2hr = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
        )

        conv1_in = in_channels + 1
        conv3_in = base_channels + 1

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, base_channels, 9, 1, 4),
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

        # output conv
        self.conv_out = nn.Sequential(
            nn.Conv2d(conv3_in, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 9, 1, 4),
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

    def forward(self, x, ext):

        ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 5].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(
            ext[:, 6].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, :4]

        ext_out = self.ext2lr(torch.cat(
            [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)

        inp = torch.cat([x, ext_out], dim=1)

        out1 = self.conv1(inp)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.cat((out1, out2), dim=1)
        out = self.conv_combin(out)

        out = self.upsampling(out)

        # concatenation backward
        ext_out = self.ext2hr(ext_out)
        out = self.conv_out(torch.cat([out, ext_out], dim=1))

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    ext = torch.rand(2, 7)
    print(input.shape)
    model = FODEE(in_channels=1,
                 out_channels=1,
                 base_channels=128)
    output = model(input, ext)
    print(output.shape)