import torch
import torch.nn as nn
import torch.nn.functional as F
from util import N2_Normalization, Recover_from_density

class ESPCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, upscale_factor=4):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, 5, 1, 2)
        self.conv2 = nn.Conv2d(base_channels, base_channels//2, 3, 1, 1)
        self.conv3 = nn.Conv2d(base_channels//2, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        # Distributional upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [nn.Conv2d(base_channels//2, base_channels//2 * 4, 3, 1, 1),
                           nn.BatchNorm2d(base_channels//2 * 4),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.den_softmax = N2_Normalization(4)
        self.recover = Recover_from_density(4)

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
    model = ESPCN(1, 1, 4)
    output = model(input)
    print(input.shape)
    print(output.shape)