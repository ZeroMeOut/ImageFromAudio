'''
Just testing this out. A generator with a classification layer that is then embedded into the noise for generation
And a normal discriminator

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size):
        super(Generator, self).__init__()

        self.classify = nn.Sequential(
            nn.Conv2d(channels_img, features_g, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_g, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_g * 8, num_classes, kernel_size=4, stride=2, padding=0),
        )

        self.embed = ()
    
    def _classify_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), 
        )

    def _gen_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

