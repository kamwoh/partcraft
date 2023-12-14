import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self,
                 rootdir,
                 filename='train.txt',
                 path_prefix='',
                 transform=None,
                 target_transform=None):
        super().__init__()

        self.rootdir = rootdir
        self.filename = filename
        self.path_prefix = path_prefix

        self.image_paths = []
        self.image_labels = []

        filename = os.path.join(self.rootdir, self.filename)

        with open(filename, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break

                lines = lines.strip()
                split_lines = lines.split(' ')
                path_tmp = split_lines[0]
                label_tmp = split_lines[1:]
                self.is_onehot = len(label_tmp) != 1
                if not self.is_onehot:
                    label_tmp = label_tmp[0]
                self.image_paths.append(path_tmp)
                self.image_labels.append(label_tmp)

        self.image_paths = np.array(self.image_paths)
        self.image_labels = np.array(self.image_labels, dtype=np.float32)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.image_paths[index], self.image_labels[index]
        target = torch.tensor(target)

        img = Image.open(f'{self.path_prefix}{path}').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.image_paths)


class DreamCreatureDataset(ImageDataset):

    def __init__(self,
                 rootdir,
                 filename='train.txt',
                 path_prefix='',
                 code_filename='train_caps.txt',
                 num_parts=8, num_k_per_part=256, repeat=1,
                 use_gt_label=False,
                 bg_code=7,
                 transform=None,
                 target_transform=None):
        super().__init__(rootdir, filename, path_prefix, transform, target_transform)

        self.image_codes = np.array(open(rootdir + '/' + code_filename).readlines())
        self.num_parts = num_parts
        self.num_k_per_part = num_k_per_part
        self.repeat = repeat
        self.use_gt_label = use_gt_label
        self.bg_code = bg_code

    def filter_by_class(self, target):
        target_mask = self.image_labels == target
        self.image_paths = self.image_paths[target_mask]
        self.image_codes = self.image_codes[target_mask]
        self.image_labels = self.image_labels[target_mask]

    def set_max_samples(self, n, seed):
        np.random.seed(seed)
        rand_idx = np.arange(len(self.image_paths))
        np.random.shuffle(rand_idx)

        self.image_paths = self.image_paths[rand_idx[:n]]
        self.image_codes = self.image_codes[rand_idx[:n]]
        self.image_labels = self.image_labels[rand_idx[:n]]

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.image_paths)
        path, target = self.image_paths[index], self.image_labels[index]
        target = torch.tensor(target)

        img = Image.open(f'{self.path_prefix}{path}').convert('RGB')

        cap = self.image_codes[index].strip()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        appeared = []

        code = torch.ones(self.num_parts) * self.num_k_per_part  # represents not exists
        splits = cap.strip().replace('.', '').split(' ')
        for c in splits:
            idx, intval = c.split(':')
            appeared.append(int(idx))
            if self.use_gt_label and self.bg_code != int(idx):
                code[int(idx)] = target
            else:
                code[int(idx)] = int(intval)

        example = {
            'pixel_values': img,
            'captions': cap,
            'codes': code,
            'labels': target,
            'appeared': appeared
        }

        return example
