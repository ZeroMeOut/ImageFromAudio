import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            self._block(channels_img, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            self._block(512, 1, 1, 1, 1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            )
            , nn.BatchNorm2d(out_channels, affine=True)
            , nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self, channels_img):
        super(Generator, self).__init__()
        # DownLayers
        self.down = nn.Sequential(
            self._downsample(channels_img, 128, 4, 2),
            self._downsample(128, 256, 4, 2),
            self._downsample(256, 512, 4, 2),
        ) 

        self.logvar_mu =  nn.Conv2d(512, 64, 4, 2, bias=False)

        # UpLayers
        self.up = nn.Sequential(
            self._upsample(64, 512, 3, 2),
            self._upsample(512, 256, 3, 2),
            self._upsample(256, 128, 2, 2),
            self._upsample(128, 1, 2, 2),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def _downsample(self, in_channels, out_channels, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),  
        )
    
    def _upsample(self, in_channels, out_channels, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  
        )

    def forward(self, x):
        h = self.down(x)

        mu, logvar = self.logvar_mu(h), self.logvar_mu(h)
        ep = torch.rand_like(logvar)
        z = logvar + mu * ep

        img = self.up(z)
        return img, mu, logvar

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Ahaha testing
# def test():
#     gen = Generator(1)
#     disc = Discriminator(1, 64)
    # x = torch.randn(32, 1, 48, 48)
    # print("gen output shape: ", gen(x).shape)
#     print("disc output shape: ", disc(gen(x)).shape)

# if __name__ == "__main__":
#     test()
