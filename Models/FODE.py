import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import N2_Normalization, Recover_from_density

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

class FODE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, n_residual_blocks=16, upscale_factor=2):
        super(FODE, self).__init__()

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

        # output conv
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        self.conv_combin = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(int(upscale_factor / 2)):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(upscale_factor)
        self.recover = Recover_from_density(upscale_factor)

    def forward(self, x):
        inp = x

        out1 = self.conv1(inp)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.cat((out1, out2), dim=1)
        out = self.conv_combin(out)

        out = self.upsampling(out)

        # concatenation backward
        out = self.conv_out(out)

        # get the distribution matrix
        out = self.den_softmax(out)
        # recover fine-grained flows from coarse-grained flows and distributions
        out = self.recover(out, x)
        return out

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    print(input.shape)
    model = FODE(in_channels=1,
                 out_channels=1,
                 base_channels=128)
    output = model(input)
    print(output.shape)