from torch import nn
import torch
from eqalized_layers import *

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block_1_from_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 1,
                out_channels = 256,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_1 = nn.Sequential(
            EqualConv2d(
                in_channels = 256,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 256,
                out_channels = 512,
                kernel_size = 3,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Flatten(),

            EqualLinear(in_features = 512, out_features = 512)
        )
        self.block_2_from_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 1,
                out_channels = 128,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_2 = nn.Sequential(
            EqualConv2d(
                in_channels = 128,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 128,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.AvgPool2d(2)
        )
        self.block_3_from_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 1,
                out_channels = 64,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2)
        )
        self.block_3 = nn.Sequential(
            EqualConv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.MaxPool2d(2)
        )
        self.block_4_from_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 1,
                out_channels = 32,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2)
        )
        self.block_4 = nn.Sequential(
            EqualConv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.MaxPool2d(2)
        )
        self.block_5_from_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2)
        )
        self.block_5 = nn.Sequential(
            EqualConv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.MaxPool2d(2),
        )
        self.average = nn.AvgPool2d(2)

    def forward(self, samples):
        output = self.block_1(self.block_2(self.block_3(
            self.block_4(self.block_5(
                self.block_5_from_rgb(samples)
            ))
        )))
        return output

class BatchStd(nn.Module):
    def __init__(self):
        super(BatchStd, self).__init__()

    def forward(self, output):
        stds = torch.unsqueeze(
            torch.std(
                output,
                dim = [0]
            ),
            0
        )
        stds = torch.mean(stds)
        stds = stds.expand(256, 1, 3, 3)
        output = torch.cat([output, stds], axis = 1)

        return output

