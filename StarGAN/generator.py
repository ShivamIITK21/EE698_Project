import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(c),
                nn.ReLU(),
                )

    def forward(self, x):
        return x + self.layer(x)

class Generator(nn.Module):
    def __init__(self, num_attributes):
        super().__init__()
        self.input_layer = nn.Sequential(
                nn.Conv2d(in_channels=num_attributes+3, out_channels=64, kernel_size=7, stride=1, padding=3),
                nn.InstanceNorm2d(64),
                nn.ReLU(),
                )

        downsampling_layers = []
        for c in [128, 256]:
            layer = nn.Sequential(
                    nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(c),
                    nn.ReLU()
                    )
            downsampling_layers.append(layer)
        self.downsample = nn.Sequential(*downsampling_layers)

        self.bottleneck = nn.Sequential(*[ResidualBlock(256) for _ in range(0, 6)])

        upsampling_layers = []
        for c in [128, 64]:
            layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=2*c, out_channels=c, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(c),
                    nn.ReLU(),
                    )
            upsampling_layers.append(layer)
        self.upsample = nn.Sequential(*upsampling_layers)

        self.output = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3),
                nn.Tanh()
                )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.downsample(x)
        x = self.bottleneck(x)
        x = self.upsample(x)
        return self.output(x)

def test_shape():
    x = torch.rand(16, 8, 256, 256)
    g = Generator(5)
    res = g(x)
    print(f"""res {res.shape}""")


if __name__ == "__main__":
    test_shape()

