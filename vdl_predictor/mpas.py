# mpas dataset

from __future__ import absolute_import, division, print_function

import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MPASDataset(Dataset):
  def __init__(self, root, direction, latent_dim, train=True, data_len=0, transform=None):
    if train:
      fh = open(os.path.join(root, "train", "names.txt"))
    else:
      fh = open(os.path.join(root, "test", "names.txt"))
    filenames = []
    for line in fh:
      filenames.append(line)

    self.root = root
    self.train = train
    self.direction = direction
    self.latent_dim = latent_dim
    self.data_len = data_len
    self.transform = transform
    self.filenames = filenames
    if self.train:
      self.params = np.load(os.path.join(root, "train/params.npy"))
    else:
      self.params = np.load(os.path.join(root, "test/params.npy"))

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
      data_name = os.path.join(self.root, "train_compressed", filename + ".npy")
    else:
      data_name = os.path.join(self.root, "test", filename + ".bin")
    depth_shape = 768
    lat_shape = 768
    lon_shape = 1536
    reduce_size = 16

    if self.train:
      data = np.load(data_name)
      if self.direction == "lon":
        assert data.shape[1] == lon_shape // reduce_size
        data = data.reshape((depth_shape // 2, lat_shape // 2, lon_shape // reduce_size, self.latent_dim))
      elif self.direction == "lat":
        assert data.shape[1] == lat_shape // reduce_size
        data = data.reshape((depth_shape // 2, lon_shape // 2, lat_shape // reduce_size, self.latent_dim))
      elif self.direction == "depth":
        assert data.shape[1] == depth_shape // reduce_size
        data = data.reshape((lat_shape // 2, lon_shape // 2, depth_shape // reduce_size, self.latent_dim))
      data = data.transpose((3, 0, 1, 2))
      mask = None
    else:
      data = np.fromfile(data_name, dtype=np.float32)
      data = data.reshape((depth_shape, lat_shape, lon_shape))
      mask = data < 10.

    params = self.params[index, 1:]
    sample = {"name": filename, "data": data, "mask": mask, "params": params}

    if self.transform:
      sample = self.transform(sample)

    return sample

# data transformation
class Normalize(object):
  def __call__(self, sample):
    name = sample["name"]
    data = sample["data"]
    mask = sample["mask"]
    params = sample["params"]

    # params min [0.0, 300.0, 0.25, 100.0]
    #        max [5.0, 1500.0, 1.0, 300.0]
    params = (params.astype(np.float32) - np.array([2.5, 900.0, .625, 200.0], dtype=np.float32)) / \
              np.array([2.5, 600.0, .375, 100.0], dtype=np.float32)

    return {"name": name, "data": data, "mask": mask, "params": params}

class ToTensor(object):
  def __call__(self, sample):
    name = sample["name"]
    data = sample["data"]
    mask = sample["mask"]
    params = sample["params"]

    return {"name": name,
            "data": torch.from_numpy(data),
            "mask": torch.from_numpy(mask),
            "params": torch.from_numpy(params)}
