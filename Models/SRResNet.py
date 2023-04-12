import torch
import torch.nn as nn
import math
from .utils import N2_Normalization, Recover_from_density
from util.data_process import print_model_parm_nums

class _Residual_Block(nn.Module):
    def __init__(self, base_channels=32):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(base_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(base_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class SRResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, upscale_factor=2):
        super(SRResNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16, base_channels)

        self.conv_mid = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(base_channels, affine=True)

        if upscale_factor == 2:
            self.upscale4x = nn.Sequential(
                nn.Conv2d(in_channels=base_channels, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(in_channels=base_channels, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.PixelShuffle(2),
                # nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale4x = nn.Sequential(
                nn.Conv2d(in_channels=base_channels, out_channels=base_channels * 4, kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=base_channels, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=9, stride=1, padding=4, bias=False)

        self.den_softmax = N2_Normalization(upscale_factor)
        self.recover = Recover_from_density(upscale_factor)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer, base_channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(base_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)

        out = self.den_softmax(out)
        out = self.recover(out, x)
        return out

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    print(input.shape)
    model = SRResNet(in_channels=1, out_channels=1, base_channels=128)
    output = model(input)
    print(output.shape)
    print_model_parm_nums(model, 'SRResNet')
