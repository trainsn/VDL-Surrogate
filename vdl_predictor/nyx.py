# nyx dataset

from __future__ import absolute_import, division, print_function

import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class NyxDataset(Dataset):
  def __init__(self, root, direction, data_size, img_size, reduce_size, latent_dim, train=True, data_len=0, transform=None):
    if train:
      fh = open(os.path.join(root, "train", "names.txt"))
    else:
      fh = open(os.path.join(root, "test", "names.txt"))
    filenames = []
    for line in fh:
      filenames.append(line)

    self.root = root
    self.direction = direction
    self.latent_dim = latent_dim
    self.train = train
    self.data_len = data_len
    self.transform = transform
    self.filenames = filenames
    if self.train:
      self.params = np.load(os.path.join(root, "train/params.npy"))
    else:
      self.params = np.load(os.path.join(root, "test/params.npy"))
    self.data_size = data_size
    self.img_size = img_size
    self.reduce_size = reduce_size

  # TODO(wenbin): deal with data_len correctly.
  def __len__(self):
    if self.data_len:
      return self.data_len
    else:
      return len(self.params)

  def __getitem__(self, index):
    if type(index) == torch.Tensor:
      index = index.item()

    filename = self.filenames[index]
    filename = filename.strip("\r\n")
    if self.train:
      data_name = os.path.join(self.root, "train_compressed", filename + ".bin")
    else:
      data_name = os.path.join(self.root, "test", filename + ".bin")
    data = np.fromfile(data_name, dtype=np.float32)

    if self.train:
      data = data.reshape((self.img_size, self.img_size, self.data_size // self.reduce_size, self.latent_dim))
      data = data.transpose((3, 0, 1, 2))
    else:
      data = np.log10(data)
      data = data.reshape((self.data_size, self.data_size, self.data_size))

    params = self.params[index, 1:]

    sample = {"name": filename, "data": data, "params": params}

    if self.transform:
      sample = self.transform(sample)

    return sample

# data transformation
class Normalize(object):
  def __call__(self, sample):
    name = sample["name"]
    data = sample["data"]
    params = sample["params"]

    # dmin = 8.6
    # dmax = 13.6
    # data = np.log10(data)
    # data = (data.astype(np.float32) - (dmax + dmin) / 2.) / ((dmax - dmin) / 2.)

    # params min [0.12, 0.0215, 0.55]
    #        max [0.155, 0.0235, 0.85]
    params = (params.astype(np.float32) - np.array([0.1375, 0.0225, 0.7], dtype=np.float32)) / \
             np.array([0.0175, 0.001, 0.15], dtype=np.float32)

    return {"name": name, "data": data, "params": params}

class ToTensor(object):
  def __call__(self, sample):
    name = sample["name"]
    data = sample["data"]
    params = sample["params"]

    return {"name": name,
            "data": torch.from_numpy(data),
            "params": torch.from_numpy(params)}
