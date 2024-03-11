from collections import defaultdict
import os
import zipfile 
import gdown
from munch import Munch
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
import numpy as np
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class CelebADataset(Dataset):
    def __init__(self,
                 args,
                 root_dir: Path = Path(CUR_DIR) / 'data/celeba',
                 transform=None,
                 remain_domains: int = 10,
                 sample_images: int = 1):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        self.args = args

        if hasattr(self.args, "root_dir") and self.args.root_dir is not None:
            root_dir = Path(self.args.root_dir)

        # Path to folder with the dataset
        if not root_dir.exists():
            root_dir.mkdir(parents=True)
        dataset_folder = str(Path(root_dir) / "img_align_celeba")
        self.dataset_folder = os.path.abspath(dataset_folder)
        if not Path(dataset_folder).exists() and not Path(root_dir / "img_align_celeba").exists():
            # URL for the CelebA dataset
            download_url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
            # Path to download the dataset to
            download_path = str(root_dir / "img_align_celeba.zip")
            # Download the dataset from google drive
            gdown.download(download_url, download_path, quiet=False)

        if not Path(root_dir / "img_align_celeba").exists():
            # Unzip the downloaded file
            download_path = str(root_dir / "img_align_celeba.zip")
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(root_dir)

        self.transform = transform

        self.filenames = []
        self.annotations = []
        with open(Path(root_dir) / "list_attr_celeba.txt") as f:
            for i, line in enumerate(f.readlines()):
                line = re.sub(' *\n', '', line)
                if i == 0:
                    self.header = re.split(' +', line)
                else:
                    values = re.split(' +', line)
                    filename = values[0]
                    self.filenames.append(filename)
                    self.annotations.append([int(v) for v in values[1:]])

        self.annotations = np.array(self.annotations)
        self.filenames = np.array(self.filenames)

        # get frequent domains
        freq_labels = np.sum(self.annotations, axis=0).argsort()[-remain_domains:]
        assert len(freq_labels) == remain_domains, f'Actual number of domains ({len(freq_labels)}) is smaller than ' \
                                                   f'provided ({remain_domains})'
        # remove rows with zero specified domains
        used_rows = np.sum(self.annotations[:, freq_labels], axis=1) != -remain_domains
        self.annotations = self.annotations[used_rows, :][:, freq_labels]
        self.filenames = self.filenames[used_rows]
        self.header = [self.header[idx] for idx in freq_labels]

        self.domain_images = []
        self.domain_to_image = defaultdict(list)
        for idx, annot in enumerate(self.annotations):
            for dom in np.where(annot > 0)[0]:
                self.domain_images.append([idx, dom])
                self.domain_to_image[dom].append(idx)

        self.sample_images = sample_images

    def __len__(self): 
        return len(self.domain_images)

    def __getitem__(self, idx):
        # Get the path to the image
        img_idx, domain = self.domain_images[idx]
        if self.sample_images > 1:
            image_indices = self.domain_to_image[domain]
            img_idx = [img_idx] + list(np.random.choice(image_indices, size=self.sample_images - 1))
        else:
            img_idx = [img_idx]

        img_names = self.filenames[img_idx]
        images = []
        for img_name in img_names:
            img_path = str(Path(self.dataset_folder) / img_name)
            # Load image and convert it to RGB
            img = Image.open(img_path).convert('RGB')
            # Apply transformations to the image
            if self.transform:
                img = self.transform(img)
            images.append(img)

        return Munch(x=images, y=domain)


class RandomDataset(Dataset):
    def __init__(self, args, size):
        self.dim = args.latent_dim
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return torch.randn(self.dim), torch.randn(self.dim)
