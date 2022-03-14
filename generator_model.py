import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, down=True, **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.residual = nn.Sequential(
            Block(in_channels, in_channels, kernel_size=3, padding=1),
            Block(in_channels, in_channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.downsample = nn.ModuleList(
            [
                Block(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                Block(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.upsample = nn.ModuleList(
            [
                Block(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                Block(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.downsample:
            x = layer(x)
        x = self.residual(x)
        for layer in self.upsample:
            x = layer(x)
        return torch.tanh(self.last(x))

def test():
    img = torch.randn(5, 3, 256, 256)
    model = Generator(img_channels=3)
    pred = model(img)
    print(pred.size())

if __name__ == "__main__":
    test()