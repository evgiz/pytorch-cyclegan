
import torch
import torch.nn as nn

import numpy as np


class ResnetBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        model = [
            nn.ReflectionPad2d(1),
            torch.nn.utils.spectral_norm(
                nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            ),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            torch.nn.utils.spectral_norm(
                nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            ),
            nn.InstanceNorm2d(dim),
        ]

        self.model = nn.Sequential(*model)
        self.dim = dim

    def __repr__(self):
        return f"ResnetBlock(dim={self.dim}, skip=2)"

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):

    def __init__(self, img_chn=3, model_chn=64, n_conv=2, n_resblock=6):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_chn, model_chn, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(model_chn),
            nn.ReLU()
        ]

        n_chn = lambda i: model_chn if i == 0 else model_chn * (2 ** i)

        # Downscale conditional
        o_chn = model_chn
        for i in range(n_conv):
            i_chn = n_chn(i)
            o_chn = n_chn(i + 1)
            model += [
                torch.nn.utils.spectral_norm(
                    nn.Conv2d(i_chn, o_chn, kernel_size=3, stride=2, padding=1),
                ),
                nn.InstanceNorm2d(o_chn),
                nn.ReLU()
            ]

        # Residual blocks
        for i in range(n_resblock):
            model += [ResnetBlock(o_chn)]

        # Upscale conditional
        for i in range(n_conv):
            i_chn = n_chn(n_conv - i)
            o_chn = n_chn(n_conv - i - 1)
            model += [
                torch.nn.utils.spectral_norm(
                    nn.ConvTranspose2d(i_chn, o_chn, kernel_size=3, stride=2, padding=1, output_padding=1),
                ),
                nn.InstanceNorm2d(o_chn),
                nn.ReLU()
            ]

        # Final convolution
        model += [
            nn.ReflectionPad2d(3),
            torch.nn.utils.spectral_norm(
                nn.Conv2d(model_chn, img_chn, kernel_size=7, padding=0),
            ),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, img_size=16, img_chn=3, model_chn=64, n_downsample=2, n_resblock=4, fc_out=False, out_act=None):
        super().__init__()

        model = []

        n_chn = lambda i: model_chn * (2 ** (i - 1))

        # Downscale
        o_chn = model_chn
        for i in range(n_downsample):
            # Channel calc
            last = i == n_downsample - 1 and not fc_out
            i_chn = img_chn if i == 0 else n_chn(i)
            o_chn = 1 if last else n_chn(i + 1)
            # Downsample
            model += [
                torch.nn.utils.spectral_norm(
                    nn.Conv2d(i_chn, o_chn, kernel_size=3, stride=2, padding=1)
                )
            ]
            if not last:
                model += [
                    nn.InstanceNorm2d(o_chn),
                    nn.LeakyReLU(0.2),
                    # nn.Dropout(0.1)
                ]
            # Residual blocks
            if i == n_downsample // 2:
                for j in range(n_resblock):
                    model += [ResnetBlock(o_chn)]

        img_size = int(img_size * 0.5 ** n_downsample)

        # Dense out
        if fc_out:
            fc_in = img_size * img_size * o_chn
            model += [
                nn.Flatten(),
                nn.Linear(fc_in, 1)
            ]
            self.output_shape = (1, )
        else:
            # PathGAN
            self.output_shape = (1, img_size, img_size)

        if out_act is not None:
            model += [out_act]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return super().__repr__() + f"\nout_shape={self.output_shape}"

    def forward(self, x):
        return self.model(x)


