import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import N2_Normalization, Recover_from_density

class SRCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, upscale_factor=2):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(base_channels, base_channels//2, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(base_channels//2, out_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(int(upscale_factor/2)):
            upsampling += [nn.Conv2d(base_channels//2, base_channels//2 * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels//2 * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(upscale_factor)
        self.recover = Recover_from_density(upscale_factor)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.upsampling(out)

        out = self.conv3(out)
        out = self.den_softmax(out)
        out = self.recover(out, x)
        return out

if __name__ == '__main__':
    input = torch.rand(2, 1, 32, 32)
    print(input.shape)
    model = SRCNN(in_channels=1,
                  out_channels=1,
                  base_channels=128)
    output = model(input)
    print(output.shape)