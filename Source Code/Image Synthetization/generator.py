from torch import nn
from eqalized_layers import *
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Linear(256, 3 * 3 * 256),
            Reshape(-1, 256, 3, 3),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),

            EqualConv2d(
                in_channels = 256,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_1_to_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 256,
                out_channels = 1,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_2 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            EqualConv2d(
                in_channels = 256,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(
                in_channels = 128,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_2_to_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels = 128,
                out_channels = 1,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_3 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            nn.Conv2d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_3_to_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 1,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_4 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_4_to_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 1,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_5 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            nn.Conv2d(
                in_channels = 32,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_5_to_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 1,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.upsample = nn.Upsample(
            scale_factor = 2,
            mode = 'bilinear'
        )

    def forward(self, samples):
        old_output = self.block_4(self.block_3(self.block_2(self.block_1(samples))))
        output = self.block_5_to_rgb(self.block_5(old_output))

        return output

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class PixelNormalization(nn.Module):
    def __init__(self):
        super(PixelNormalization, self).__init__()

    def forward(self, output):
        return output / torch.sqrt(
            torch.unsqueeze(
                torch.mean(
                    torch.pow(output, 2),
                    dim = [1],
                ) + 1e-8,
                1
            )
        )

