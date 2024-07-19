"""
The `ConvBlock` class is a PyTorch module that represents a convolutional block with optional downsampling, instance normalization, and activation.

Args:
    in_channels (int): The number of input channels.
    out_channels (int): The number of output channels.
    down (bool, optional): Whether to use a convolutional layer or a transposed convolutional layer. Defaults to `True`.
    use_act (bool, optional): Whether to use an activation function. Defaults to `True`.
    **kwargs: Additional keyword arguments passed to the convolutional layer.

Attributes:
    conv (nn.Sequential): The convolutional block.

Methods:
    forward(x: torch.Tensor) -> torch.Tensor:
        Applies the convolutional block to the input tensor `x`.
"""

"""
The `ResidualBlock` class is a PyTorch module that represents a residual block, which consists of two convolutional blocks with a skip connection.

Args:
    channels (int): The number of channels in the input and output tensors.

Attributes:
    block (nn.Sequential): The residual block.

Methods:
    forward(x: torch.Tensor) -> torch.Tensor:
        Applies the residual block to the input tensor `x`.
"""

"""
The `Generator` class is a PyTorch module that represents a generator network for a generative adversarial network (GAN).

Args:
    img_channels (int): The number of channels in the input and output images.
    num_features (int, optional): The number of features in the initial convolutional layer. Defaults to 64.
    num_residuals (int, optional): The number of residual blocks. Defaults to 9.

Attributes:
    initial (nn.Sequential): The initial convolutional layer with instance normalization and ReLU activation.
    down_blocks (nn.ModuleList): The downsampling convolutional blocks.
    res_blocks (nn.Sequential): The residual blocks.
    up_blocks (nn.ModuleList): The upsampling convolutional blocks.
    last (nn.Conv2d): The final convolutional layer that outputs the generated image.

Methods:
    forward(x: torch.Tensor) -> torch.Tensor:
        Applies the generator network to the input tensor `x` and returns the generated image.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()