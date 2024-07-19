"""
The `Discriminator` class is a PyTorch module that implements a convolutional neural network for discriminating between real and fake images. It takes an input image and outputs a scalar value between 0 and 1, representing the probability that the input is a real image.

The `Discriminator` class consists of several `Block` modules, each of which applies a convolutional layer, instance normalization, and a leaky ReLU activation function. The initial layer applies a convolutional layer and a leaky ReLU activation function. The final layer applies a convolutional layer with a sigmoid activation function to produce the output.

The `forward` method takes an input image and returns the probability that the input is a real image.
"""
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0,2),
        )

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()