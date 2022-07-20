# MPAS dataset

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SeqDataset(Dataset):
    def __init__(self, root, data_size, img_size, direction, train=True, data_len=0, transform=None):
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
        self.data_size = data_size
        self.img_size = img_size

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


        data = np.fromfile(data_name, dtype=np.float32)
        if self.direction == "x":
            data = data.reshape((self.img_size, self.img_size, self.data_size))
        if self.direction == "y":
            data = data.reshape((self.img_size, self.data_size, self.img_size))
            data = np.transpose(data, (0, 2, 1))
        elif self.direction == "z":
            data = data.reshape((self.data_size, self.img_size, self.img_size))
            data = np.transpose(data, (1, 2, 0))

        sample = {"name": filename, "data": data}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalize(object):
    def __call__(self, sample):
        name = sample["name"]
        data = sample["data"]
        dmin = 8.6
        dmax = 13.6

        data = np.log10(data)
        data = (data.astype(np.float32) - (dmax + dmin) / 2.) / ((dmax - dmin) / 2.)

        return {"name": name, "data": data}

class ToTensor(object):
    def __call__(self, sample):
        name = sample["name"]
        data = sample["data"]

        # dimension raising
        # numpy shape: [N, ]
        # torch shape: [1, N]
        data = data[None, :]
        return {"name": name,
                "data": torch.from_numpy(data)}
