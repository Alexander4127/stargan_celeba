import copy
from functools import partial
from itertools import chain
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, downsample=False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        if downsample:
            self.downsampler = partial(F.avg_pool2d, kernel_size=2)
        else:
            self.downsampler = nn.Identity()

        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)

        if dim_in != dim_out:
            self.resampler = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        else:
            self.resampler = nn.Identity()

    def forward(self, x):
        out1 = self.downsampler(self.resampler(x))
        out2 = self.downsampler(self.conv1(self.actv(self.norm1(x))))
        out2 = self.conv2(self.actv(self.norm2(out2)))
        return (out1 + out2) / math.sqrt(2)


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaptiveResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, upsample=False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        if upsample:
            self.upsampler = partial(F.interpolate, scale_factor=2, mode='nearest')
        else:
            self.upsampler = nn.Identity()

        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_in)

        if dim_in != dim_out:
            self.resampler = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        else:
            self.resampler = nn.Identity()

    def forward(self, x, s):
        out1 = self.resampler(self.upsampler(x))
        out2 = self.upsampler(self.conv1(self.actv(self.norm1(x, s))))
        out2 = self.conv2(self.actv(self.norm2(out2, s)))
        return (out1 + out2) / math.sqrt(2)


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, **kwargs):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        assert repeat_num > 0
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(ResBlock(dim_in, dim_out, downsample=True))
            self.decode.insert(0, AdaptiveResBlock(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlock(dim_out, dim_out))
            self.decode.insert(0, AdaptiveResBlock(dim_out, dim_out, style_dim))

    def forward(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, hidden_size=512, num_domains=2, **kwargs):
        super().__init__()
        self.shared = self._create_block(hidden_size, input_dim=latent_dim)
        self.unshared = nn.ModuleList([self._create_block(hidden_size, output_dim=style_dim) for _ in range(num_domains)])

    @staticmethod
    def _create_block(inter_dim, input_dim=None, output_dim=None, num_layers=4):
        return nn.Sequential(
            *chain(*[
                [nn.Linear(
                    input_dim if idx == 0 and input_dim is not None else inter_dim,
                    output_dim if idx + 1 == num_layers and output_dim is not None else inter_dim
                ), nn.LeakyReLU(0.2)]
                for idx in range(num_layers)
            ])
        )

    def forward(self, z, y):
        h = self.shared(z)
        out = torch.stack([layer(h) for layer in self.unshared], dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512, **kwargs):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        assert repeat_num > 0
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 4, 1, 0), nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList([nn.Linear(dim_out, style_dim) for _ in range(num_domains)])

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, **kwargs):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        assert repeat_num > 0
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2), nn.Conv2d(dim_out, num_domains, 1, 1, 0)]

        self.body = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.body(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        print(out.shape, idx, y)
        out = out[idx, y]  # (batch)
        return out


def build_model(device, **kwargs):
    generator = Generator(**kwargs).to(device)
    generator_ema = copy.deepcopy(generator).to(device)
    mapping_network = MappingNetwork(**kwargs).to(device)
    mapping_network_ema = copy.deepcopy(mapping_network).to(device)
    style_encoder = StyleEncoder(**kwargs).to(device)
    style_encoder_ema = copy.deepcopy(style_encoder).to(device)

    # "for evaluation, we employ exponential moving averages
    # over parameters of all modules except D [discriminator]"
    discriminator = Discriminator(**kwargs).to(device)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)

    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema
