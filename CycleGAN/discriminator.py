import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, padding_mode="reflect"),
                nn.LeakyReLU(0.2),
                )


        self.mid_layers = []
        for i in [128, 256, 512]:
            conv_block = nn.Sequential(
                    nn.Conv2d(i//2, i, 4, 2 if i != 512 else 1, 1, padding_mode="reflect"),
                    nn.InstanceNorm2d(i),
                    nn.LeakyReLU(0.2)
                    )
            self.mid_layers.append(conv_block)
        self.bottleneck = nn.Sequential(*self.mid_layers)
        self.last_conv = nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect")


    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bottleneck(x)
        x = torch.sigmoid(self.last_conv(x))
        return x

def test_shape():
    x = torch.rand(5, 3, 256, 256)
    d = Discriminator()
    res = d(x)
    print(f"""res {res.shape}""")


if __name__ == "__main__":
    test_shape()
