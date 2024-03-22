"""
This module implements a U-Net architecture for image segmentation tasks.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with ReLU and BatchNorm.
    Used in U-Net's down path for feature extraction.

    Note:
        "same" padding is used in convolutions.

    Args:
        in_channels (int): Number of input channels for the first convolution.
        out_channels (int): Number of output channels for the second convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNET(nn.Module):
    """
    A U-Net architecture for image segmentation tasks.

    Note:
        "same" padding is used in convolutions to preserve feature map size for
        upsampling skip connections.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        features (list of int, optional): A list of feature channel counts for each
                                          downsampling step in the U-Net.
                                          Defaults to [64, 128, 256, 512].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Tuple[int] = (64, 128, 256, 512),
    ) -> None:

        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], 2 * features[-1])

        # Up part of U-Net
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(2 * feature, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), axis=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


if __name__ == "__main__":
    model = UNET()
    random_input = torch.randn((1, 3, 100, 100))
    output = model(random_input)
    print(f"{output.shape = }")
