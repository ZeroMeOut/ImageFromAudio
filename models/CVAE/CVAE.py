import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

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
        
        self.fc1 = nn.Linear(features_g * 8, 10)
        self.fc2 = nn.Linear(features_g * 8, 10)

        # UpLayers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(10, features_g * 4, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._upsample(features_g * 4, features_g * 2, 2, 2, 1),
            self._upsample(features_g * 2, features_g, 2, 2, 1),
            nn.ConvTranspose2d(features_g, channels_img, 1, 3, 0, bias=False), # I know this is weird but it works ( I calculated it to bring out 28x28 image)
        )

        self.relu = nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.ReLU(),  
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

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(self.device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottelneck(self, h):
        h = h.view(h.shape[0], -1)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        return mu, logvar, z
    
    def forward(self, x):
        # Encode
        h = self.down(x)

        mu, logvar, z = self.bottelneck(h)
        z = self.up(z)
        return z, mu, logvar


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy_with_logits(recon_x, x)
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + 0.0001*KLD, BCE, KLD # From this paper 

# Ahaha testing
# def test():
#     gen = Generator(1, 64)

    # summary(gen, input_size=(32, 1, 28, 28))
    # initialize_weights(gen)

    # x = torch.randn(32, 1, 28, 28)
    # z, mu, logvar = gen(x)
    # print("gen output shape: ", z.shape, mu.shape, logvar.shape)
    # # loss, bce, kld = loss_fn(z, y, mu, logvar)
    # # print(loss, bce, kld)


# if __name__ == "__main__":
#     test()
