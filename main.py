import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm.auto import trange
import wandb

from lpips_pytorch import LPIPS
from torchvision import transforms
from munch import Munch

from celeba import CelebADataset, RandomDataset
from solver import Solver


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model params
    args.latent_dim = 16
    args.hidden_dim = 512
    args.style_dim = 64
    args.img_size = 256

    # Transformations to be applied to each individual image sample
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    # Load the dataset from file and apply transformations
    dataset = CelebADataset(args, transform=transform)
    ref_dataset = CelebADataset(args, transform=transform, sample_images=2)
    rand_dataset = RandomDataset(args, size=len(dataset))
    args.data = [dataset, ref_dataset, rand_dataset]

    args.num_domains = len(dataset.header)

    # Number of workers for the dataloader
    num_workers = 0 if device.type == 'cuda' else 2
    # Whether to put fetched data tensors to pinned memory
    pin_memory = True if device.type == 'cuda' else False

    # dataloader for batched data loading
    loader = torch.utils.data.DataLoader(
        dataset,  batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True
    )

    ref_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True
    )

    rand_loader = torch.utils.data.DataLoader(
        rand_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True
    )

    args.loaders = [loader, ref_loader, rand_loader]

    # argument setup w. r. t. the paper

    # optimizer params
    args.mode = "train"
    args.lr = 1e-4
    args.f_lr = 1e-6
    args.beta1 = 0.0
    args.beta2 = 0.99
    args.weight_decay = 1e-4

    # dirs
    args.checkpoint_dir = "checkpoints"
    args.sample_dir = "samples"

    # iters
    args.resume_iter = 0
    args.total_iter = 100

    # logging
    args.print_every = 100
    args.sample_every = 100
    args.save_every = 1000

    # loss coefs
    args.lambda_sty = 1
    args.lambda_ds = 1
    args.lambda_cyc = 1

    Solver(args).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
