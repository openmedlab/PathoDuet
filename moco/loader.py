# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
import cv2


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
class TCGADataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 used_TCGA, 
                 transform=None):
        # data storage: *data_dir*/TCGA-xxxx/region(patch)/xxxxx.png
        self.data_paths = []
        for tcga in used_TCGA:
            data_prefix = os.path.join(data_dir, tcga)
            patch_paths = [os.path.join(data_prefix, 'patch', d) for d in os.listdir(os.path.join(data_prefix, 'patch'))]
            region_paths = [os.path.join(data_prefix, 'region', d) for d in  os.listdir(os.path.join(data_prefix, 'region'))]
            self.data_paths += patch_paths + region_paths
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.data_paths)


class BridgeDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 used_TCGA, 
                 patch_transform, 
                 region_transform):
        
        # data_dir: *TCGA_crop*/TCGA-xxxx/region(patch)/xxxxx.png

        self.data_dir = data_dir
        self.patch_name_list = []
        for tcga in used_TCGA:
            data_prefix = os.path.join(data_dir, tcga)
            patch_paths = [os.path.join(tcga, 'patch', d) for d in os.listdir(os.path.join(data_prefix, 'patch'))]
            self.patch_name_list += patch_paths

        self.patch_transform = patch_transform
        self.region_transform = region_transform
        self.ref_transform = transforms.Compose([
            transforms.Resize(16), 
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        patch_name = self.patch_name_list[idx]  # TCGA-xxxx/patch/a_b_c.png c=0, 1, ..., 8
        region_name = patch_name[:-6] + '.png'  # TCGA-xxxx/region/a_b.png
        region_name = region_name.replace('patch', 'region')
        patch_path = self.data_dir + patch_name
        region_path = self.data_dir + region_name
        
        # Loading images
        patch = Image.open(patch_path).convert('RGB')
        region = Image.open(region_path).convert('RGB')
        
        patchs = self.patch_transform(patch)
        region = self.region_transform(region)
        ref = self.ref_transform(patch)
        
        return patchs + [ref, region]

    def __len__(self):
        return len(self.patch_name_list)
