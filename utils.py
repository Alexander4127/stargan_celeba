import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import wandb


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return np.transpose(out.clamp_(0, 1).cpu().numpy(), axes=(1, 2, 0))


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref
    z_trg = inputs.z_trg

    style = nets.style_encoder(x_ref, y_ref)
    x = nets.generator(x_src, style)
    style_latent = nets.mapping_network(z_trg, y_ref)
    x_latent = nets.generator(x_src, style_latent)

    plt.gcf().set_size_inches(9, 9)

    plt.subplot(2, 2, 1)
    plt.title(f"Initial: domain = {args.datasets[0].get_label(y_src[0])}")
    plt.imshow(denormalize(x_src[0]))

    plt.subplot(2, 2, 2)
    plt.title(f"Reference: domain = {args.datasets[0].get_label(y_ref[0])}")
    plt.imshow(denormalize(x_ref[0]))

    plt.subplot(2, 2, 3)
    plt.title(f"Generated: domain = {args.datasets[0].get_label(y_ref[0])}")
    plt.imshow(denormalize(x[0]))

    plt.subplot(2, 2, 4)
    plt.title(f"Generated (lat): domain = {args.datasets[0].get_label(y_ref[0])}")
    plt.imshow(denormalize(x_latent[0]))

    if args.use_wandb:
        wandb.log({"img": plt}, step=step)

    return plt

