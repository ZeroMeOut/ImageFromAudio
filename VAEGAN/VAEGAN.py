import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, 2, 2, 1),
            self._block(features_d * 2, features_d * 4, 2, 2, 1),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 0, bias=False),
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

        self.sigma_mu =  nn.Conv2d(512, 64, 4, 2, bias=False)

        # UpLayers
        self.up = nn.Sequential(
            self._upsample(64, 512, 3, 2),
            self._upsample(512, 256, 3, 2),
            self._upsample(256, 128, 2, 2),
            self._upsample(128, 1, 2, 2),
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

        mu, sigma = self.sigma_mu(h), self.sigma_mu(h)
        ep = torch.rand_like(sigma)
        z = sigma + mu * ep

        img = self.up(z)
        return img, mu, sigma

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
