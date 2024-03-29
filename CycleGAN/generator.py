import torch.nn as nn
import torch

class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.l1 = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(channels),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(channels),
                nn.ReLU(),
                )

    def forward(self, x):
        return x + self.l1(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
                nn.InstanceNorm2d(64),
                nn.ReLU(),
                )


        self.downsampling_layers = []
        for c in [128, 256]:
            layer = nn.Sequential(
                    nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                    nn.InstanceNorm2d(c),
                    nn.ReLU(),
                    )
            self.downsampling_layers.append(layer)
        self.downsample = nn.Sequential(*self.downsampling_layers)

        self.residual = nn.Sequential(*[ResidualConv(256) for _ in range(0, 9)])

        self.upsampling_layers = []
        for c in [128, 64]:
            layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(c),
                    nn.ReLU(),
                    )
            self.upsampling_layers.append(layer)
        self.upsample = nn.Sequential(*self.upsampling_layers)

        self.final = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        return torch.tanh(self.final(x))


def test_shape():
    x = torch.rand(6, 3, 256, 256)
    g = Generator()
    res = g(x)
    print(f"""res {res.shape}""")


if __name__ == "__main__":
    test_shape()
