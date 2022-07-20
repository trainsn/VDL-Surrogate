# MPAS dataset

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SeqDataset(Dataset):
    def __init__(self, root, direction,  train=True, data_len=0, transform=None):
        if train:
            fh = open(os.path.join(root, "train", "names0.txt"))
        else:
            fh = open(os.path.join(root, "train", "names.txt"))
        filenames = []
        for line in fh:
            filenames.append(line)

        self.root = root
        self.direction = direction
        self.train = train
        self.data_len = data_len
        self.transform = transform
        self.filenames = filenames
        self.depth_shape = 768
        self.lat_shape = 768
        self.lon_shape = 1536

    def __len__(self):
        if self.data_len:
            return self.data_len
        else:
            return len(self.filenames)

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()

        filename = self.filenames[index]
        filename = filename.strip("\r\n")
        if self.train:
            data_name = os.path.join(self.root, "train", filename + ".bin")
        else:
            data_name = os.path.join(self.root, "train", filename + ".bin")

        data = np.fromfile(data_name, dtype=np.float32).reshape((self.depth_shape, self.lat_shape, self.lon_shape))
        if self.direction == "lon":
            data = data.reshape((-1, self.lon_shape))
        if self.direction == "lat":
            data = np.transpose(data, (0, 2, 1))
            data = data.reshape((-1, self.lat_shape))
        elif self.direction == "depth":
            data = np.transpose(data, (1, 2, 0))
            data = data.reshape((-1, self.depth_shape))
        mask = data < 10.

        sample = {"name": filename, "data": data, "mask": mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        name = sample["name"]
        data = sample["data"]
        mask = sample["mask"]

        dmin = 10.09
        dmax = 29.85
        data = (data.astype(np.float32) - (dmax + dmin) / 2.) / ((dmax - dmin) / 2.)

        return {"name": name, "data": data, "mask": mask}

class ToTensor(object):
    def __call__(self, sample):
        name = sample["name"]
        data = sample["data"]
        mask = sample["mask"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [1, N]
        data = data[None, :]
        mask = mask[None, :]
        assert data.shape[0] == 1
        return {"name": name,
                "data": torch.from_numpy(data),
                "mask": torch.from_numpy(mask)}
