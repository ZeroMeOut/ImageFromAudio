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
    def __init__(self, channels_img, features_g):
        super(Generator, self).__init__()
        # DownLayers
        self.down = nn.Sequential(
            nn.Conv2d(channels_img, features_g, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._downsample(features_g, features_g * 2, 2, 2, 1),
            self._downsample(features_g * 2, features_g * 4, 2, 2, 1),
            nn.Conv2d(features_g * 4, features_g * 8, 4, 2, 0, bias=False),
        )
        
        self.img_2hid = nn.Linear(features_g * 8, features_g * 4)
        self.hid_2mu = nn.Linear(features_g * 4, 10)
        self.hid_2sigma = nn.Linear(features_g * 4, 10)
        self.z_2hid = nn.Linear(10, features_g * 4)
        self.hid_2img = nn.Linear(features_g * 4, features_g * 8)

        # UpLayers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._upsample(features_g * 4, features_g * 2, 2, 2, 1),
            self._upsample(features_g * 2, features_g, 2, 2, 1),
            nn.ConvTranspose2d(features_g, channels_img, 1, 3, 0, bias=False), # I know this is weird but it works ( I calculated it  to bring out 28x28 image)
        )

        self.relu = nn.ReLU()

    def _downsample(self, in_channels, out_channels, kernel_size, stride, padding):
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
            nn.ReLU(),  
        )
    
    def _upsample(self, in_channels, out_channels, kernel_size, stride, padding):
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

        h = h.view(h.shape[0], -1)
        h = self.img_2hid(h)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        ep = torch.rand_like(sigma)
        z = sigma + mu * ep
        z = self.z_2hid(z)
        z = self.hid_2img(z)

        z_unsqueeze = z.view(z.shape[0], z.shape[1], 1, 1)

        img = self.up(z_unsqueeze)
        return img

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Ahaha testing
# def test():
#     gen = Generator(1, 64)
#     disc = Discriminator(1, 64)
#     x = torch.randn(32, 1, 28, 28)
#     print("gen output shape: ", gen(x).shape)
#     print("disc output shape: ", disc(gen(x)).shape)

# if __name__ == "__main__":
#     test()
