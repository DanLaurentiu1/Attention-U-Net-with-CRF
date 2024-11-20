import torch
import torchvision.transforms.functional
from torch import nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.first(x)
        x = self.batch_norm(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)
