import torch
import torch.nn as nn
from math import sqrt
from .utils import N2_Normalization, Recover_from_density
from util.data_process import print_model_parm_nums

class Conv_ReLU_Block(nn.Module):
    def __init__(self, base_channels=128):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=128, upscale_factor=2):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18, base_channels)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        upsampling = []
        for out_features in range(int(upscale_factor/2)):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(upscale_factor)
        self.recover = Recover_from_density(upscale_factor)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer, base_channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(base_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu(self.input(x))
        out = self.residual_layer(out)

        out = self.upsampling(out)

        out = self.output(out)

        out = self.den_softmax(out)
        out = self.recover(out, x)

        return out

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    # print(input.shape)
    # model = VDSR(in_channels=1, out_channels=1, base_channels=128)
    # output = model(input)
    # print(output.shape)
    # print_model_parm_nums(model, 'VDSR')