# model evaluation

from __future__ import absolute_import, division, print_function

import os
import argparse
import math

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from mpas import *
from aemodel import *
from generator import Generator

import pdb

# parse arguments
def parse_args():
  parser = argparse.ArgumentParser(description="Deep Learning Model")

  parser.add_argument("--no-cuda", action="store_true", default=False,
                      help="disables CUDA training")
  parser.add_argument("--data-parallel", action="store_true", default=False,
                      help="enable data parallelism")
  parser.add_argument("--seed", type=int, default=1,
                      help="random seed (default: 1)")

  parser.add_argument("--root", required=True, type=str,
                      help="root of the dataset")
  parser.add_argument("--resume", type=str, default="",
                      help="path to the latest checkpoint (default: none)")
  parser.add_argument("--ae-resume", required=True, type=str,
                      help="path to AutoEncoder")

  parser.add_argument("--dsp", type=int, default=4,
                      help="dimensions of the simulation parameters (default: 4)")
  parser.add_argument("--dspe", type=int, default=512,
                      help="dimensions of the simulation parameters' encode (default: 512)")
  parser.add_argument("--latent-dim", type=int, default=3,
                      help="latent vector dimension(default: 3)")
  parser.add_argument("--ch", type=int, default=64,
                      help="channel multiplier of network (default: 64)")
  parser.add_argument("--ae-ch", type=int, default=64,
                      help="channel multiplier of AutoEncoder (default: 64)")
  parser.add_argument("--direction", type=str, required=True,
                      help="parallel direction: lon, lat or depth")

  parser.add_argument("--sn", action="store_true", default=False,
                      help="enable spectral normalization")

  parser.add_argument("--batch-size", type=int, default=50,
                      help="batch size for training (default: 50)")
  parser.add_argument("--start-epoch", type=int, default=0,
                      help="start epoch number (default: 0)")

  parser.add_argument("--id", type=int, default=0,
                      help="index of the data to evaluate (default: 0)")
  parser.add_argument("--save", action="store_true", default=False,
                      help="save the npy file")

  return parser.parse_args()

# the main function
def main(args):
  # log hyperparameters
  print(args)

  # select device
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda:0" if args.cuda else "cpu")

  # set random seed
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  # data loader
  train_dataset = MPASDataset(
      root=args.root,
      direction=args.direction,
      latent_dim=args.latent_dim,
      train=True,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  test_dataset = MPASDataset(
      root=args.root,
      direction=args.direction,
      latent_dim=args.latent_dim,
      train=False,
      transform=transforms.Compose([Normalize(), ToTensor()]))

  kwargs = {"num_workers": 2, "pin_memory": True} if args.cuda else {}
  # train_loader = DataLoader(train_dataset, batch_size=1,
  #                           shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=1,
                           shuffle=False, **kwargs)

  # model
  def weights_init(m):
    if isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def add_sn(m):
    for name, c in m.named_children():
      m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv1d)):
      return nn.utils.spectral_norm(m, eps=1e-4)
    else:
      return m

  g_model = Generator(args.direction, args.latent_dim, args.dsp, args.dspe, args.ch)
  g_model.apply(weights_init)
  g_model.to(device)
  decoder = Decoder(args.latent_dim, args.ae_ch)
  if args.sn:
      decoder = add_sn(decoder)
  decoder.to(device)

  # loss
  mse_criterion = nn.MSELoss(reduction="none").cuda()

  # load checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint {}".format(args.resume))
      checkpoint = torch.load(args.resume, map_location=device)
      args.start_epoch = checkpoint["epoch"]
      g_model.load_state_dict(checkpoint["g_model_state_dict"])
      # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
      print("=> loaded checkpoint {} (epoch {})"
          .format(args.resume, checkpoint["epoch"]))

  if args.ae_resume:
      if os.path.isfile(args.ae_resume):
          print("=> loading checkpoint {}".format(args.ae_resume))
      checkpoint = torch.load(args.ae_resume, map_location='cpu')
      args.start_epoch = checkpoint["epoch"]
      decoder.load_state_dict(checkpoint["decoder_state_dict"])
      print("=> loaded checkpoint {} (epoch {})"
            .format(args.ae_resume, checkpoint["epoch"]))
      del checkpoint

  decoder.eval()
  # In BatchNorm, we still want the mean and var calculated from the current instance
  g_model.train()

  mse = 0.
  psnrs = np.zeros(len(test_loader.dataset))
  max_diff = np.zeros(len(test_loader.dataset))

  log = np.zeros((len(test_loader.dataset), 3))

  dmin = 10.09
  dmax = 29.85

  depth_shape = 768
  lat_shape = 768
  lon_shape = 1536
  reduce_size = 16

  with torch.no_grad():
    for i, sample in enumerate(test_loader):
        name = sample["name"][0]
        simIdx = int(name[:name.find("_")])
        sparams = sample["params"].cuda(non_blocking=True)
        fake = g_model(sparams)

        data = sample["data"][0].cuda(non_blocking=True)
        mask = sample["mask"][0].cuda(non_blocking=True)
        fake_compressed_data = fake[0]
        fake_compressed_data = fake_compressed_data.permute((1, 2, 0, 3))
        if args.direction == "lon":
            fake_compressed_data = fake_compressed_data.view((-1, args.latent_dim, lon_shape // reduce_size))
            fake_data = -1. * torch.ones((lat_shape * depth_shape // 4, lon_shape)).cuda(0)
            num_batch = (lat_shape * depth_shape // 4 - 1) // args.batch_size + 1
            indices = np.arange(lat_shape * depth_shape // 4)
        elif args.direction == "lat":
            fake_compressed_data = fake_compressed_data.view((-1, args.latent_dim, lat_shape // reduce_size))
            fake_data = -1. * torch.ones((lon_shape * depth_shape // 4, lat_shape)).cuda(0)
            num_batch = (lon_shape * depth_shape // 4 - 1) // args.batch_size + 1
            indices = np.arange(lon_shape * depth_shape // 4)
        elif args.direction == "depth":
            fake_compressed_data = fake_compressed_data.view((-1, args.latent_dim, depth_shape // reduce_size))
            fake_data = -1. * torch.ones((lon_shape * lat_shape // 4, depth_shape)).cuda(0)
            num_batch = (lon_shape * lat_shape // 4 - 1) // args.batch_size + 1
            indices = np.arange(lon_shape * lat_shape // 4)

        for j in range(num_batch):
            if j == num_batch - 1:
                sub_compressed = fake_compressed_data[indices[j * args.batch_size: indices.shape[0]]]
            else:
                sub_compressed = fake_compressed_data[indices[j * args.batch_size: (j + 1) * args.batch_size]]
            sub_recon = decoder(sub_compressed).squeeze(1)
            if j == num_batch - 1:
                fake_data[indices[j * args.batch_size: indices.shape[0]]] = sub_recon
            else:
                fake_data[indices[j * args.batch_size: (j + 1) * args.batch_size]] = sub_recon
            del sub_compressed, sub_recon
        fake_data = fake_data * (dmax - dmin) / 2 + (dmax + dmin) / 2.
        if args.direction == "lon":
            fake_data = fake_data.view((1, 1, depth_shape // 2, lat_shape // 2, lon_shape))
            fake_data = F.interpolate(fake_data, scale_factor=[2., 2., 1.], mode="trilinear")[0, 0]
        elif args.direction == "lat":
            fake_data = fake_data.view((1, 1, depth_shape // 2, lon_shape // 2, lat_shape))
            fake_data = fake_data.permute((0, 1, 2, 4, 3))
            fake_data = F.interpolate(fake_data, scale_factor=[2., 1., 2.], mode="trilinear")[0, 0]
        elif args.direction == "depth":
            fake_data = fake_data.view((1, 1, lat_shape // 2, lon_shape // 2, depth_shape))
            fake_data = fake_data.permute((0, 1, 4, 2, 3))
            fake_data = F.interpolate(fake_data, scale_factor=[1., 2., 2.], mode="trilinear")[0, 0]

        diff = abs((data - fake_data) * ~mask)
        max_diff[i] = diff.max().item() / (dmax - dmin)
        loss = (mse_criterion(data, fake_data) * ~mask).sum() / (~mask).sum()
        mse += loss.item()
        psnrs[i] = 20. * np.log10(dmax - dmin) - 10. * np.log10(loss.item())
        del data

        log[i] = np.array([simIdx, psnrs[i], max_diff[i]])

        print("{:d},{:4f},{:4f}".format(simIdx, psnrs[i], max_diff[i]))

        if args.save:
            fake_data[mask] = -32767
            output = fake_data.cpu().numpy()
            output.tofile(os.path.join(args.root, "pred", "{}.bin".format(name)))
            del output
        del fake_data, fake

  print("====> PSNR on raw {}"
        .format(20. * np.log10(dmax - dmin) -
                10. * np.log10(mse / len(test_dataset))))
  print("====> max difference on raw {}"
        .format(max_diff.mean()))

  np.save(os.path.join(args.root, "log"), log)

if __name__ == "__main__":
  main(parse_args())
