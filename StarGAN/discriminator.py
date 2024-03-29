import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, image_dim, num_attributes):
        super().__init__()
        self.input_layer = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01)
                )
        
        hidden_layers = []
        for c in [128, 256, 512, 1024, 2048]:
            layer = nn.Sequential(
                    nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.01)
                    )
            hidden_layers.append(layer)
        self.hidden = nn.Sequential(*hidden_layers)
    
        self.d_src = nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.d_class = nn.Conv2d(in_channels=2048, out_channels=num_attributes, kernel_size=image_dim//64,stride=1, padding=0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden(x)
        return self.d_src(x), self.d_class(x)

def test_shape():
    x = torch.rand(16, 3, 128, 128)
    d = Discriminator(128, 5)
    d_src, d_class = d(x)
    print(f"""d_src {d_src.shape}""")
    print(f"""d_class {d_class.shape}""")
        
if __name__ == "__main__":
    test_shape()
