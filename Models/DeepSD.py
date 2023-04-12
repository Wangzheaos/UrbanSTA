import torch
import torch.nn as nn
import torch.nn.functional as F
from util import N2_Normalization, Recover_from_density

class DeepSD(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(DeepSD, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels//2, kernel_size=5, padding=5//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=5, padding=5 // 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels//2, kernel_size=5, padding=5//2),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Conv2d(base_channels // 2, out_channels, kernel_size=5, padding=5 // 2)

        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(base_channels // 2, base_channels // 2 * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels // 2 * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.upsampling(out)

        out = self.conv_out(out)
        out = self.den_softmax(out)
        out = self.recover(out, x)
        return out

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    print(input.shape)
    model = DeepSD(in_channels=1,
                  out_channels=1,
                  base_channels=128)
    output = model(input)
    print(output.shape)