import pdb

import os
import argparse
import math

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from seq import *
from aemodel import *

from apex import amp

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--latent-dim", type=int, default=3,
                        help="latent vector dimension(default: 3)")
    parser.add_argument("--ch", type=int, default=64,
                        help="channel multiplier (default: 64)")
    parser.add_argument("--direction", type=str, required=True,
                        help="parallel direction: depth, lat or lon")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--opt-level", default='O2',
                        help='amp opt_level, default="O2"')
    parser.add_argument("--save", action="store_true", default=False,
                        help="save the npy file")

    return parser.parse_args()

# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data loader
    train_dataset = SeqDataset(
        root=args.root,
        direction=args.direction,
        train=True,
        transform=transforms.Compose([Normalize(), ToTensor()]))
    test_dataset = SeqDataset(
        root=args.root,
        direction=args.direction,
        train=False,
        transform=transforms.Compose([Normalize(), ToTensor()]))

    kwargs = {"num_workers": 2, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1,
                              shuffle=False, **kwargs)

    # model
    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    encoder = Encoder(args.latent_dim, args.ch)
    if args.sn:
        encoder = add_sn(encoder)
    encoder.cuda()
    decoder = Decoder(args.latent_dim, args.ch)
    if args.sn:
        decoder = add_sn(decoder)
    decoder.cuda()

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint["epoch"]
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        train_losses = checkpoint["train_losses"]
        test_losses = checkpoint["test_losses"]
        print("=> loaded checkpoint {} (epoch {})"
              .format(args.resume, checkpoint["epoch"]))
        del checkpoint

    mse_criterion = nn.MSELoss(reduction="none").cuda()

    reduce_size = 16

    dmin = 10.09
    dmax = 29.85

    mse = 0.
    psnrs = np.zeros(len(test_loader.dataset))
    max_diff = np.zeros(len(test_loader.dataset))

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        num_seq = 0
        for i, sample in enumerate(test_loader):
            data = sample["data"][0]
            mask = sample["mask"][0]
            data = data.reshape((-1, data.shape[-1])).cuda(non_blocking=True)
            mask = mask.reshape((-1, mask.shape[-1])).cuda(non_blocking=True)
            data = data.unsqueeze(1)
            mask = mask.unsqueeze(1)
            indices = torch.arange(data.shape[0])
            num_seq += indices.shape[0]

            compressed = torch.zeros((data.shape[0], args.latent_dim, data.shape[-1] // reduce_size)).cuda()
            recon = torch.zeros((data.shape[0], data.shape[-1])).cuda()

            num_batch = (indices.shape[0] - 1) // args.batch_size + 1
            for j in range(num_batch):
                if j == num_batch - 1:
                    sub_data = data[indices[j * args.batch_size: data.shape[0]]]
                    sub_mask = mask[indices[j * args.batch_size: data.shape[0]]]
                else:
                    sub_data = data[indices[j * args.batch_size: (j + 1) * args.batch_size]]
                    sub_mask = mask[indices[j * args.batch_size: (j + 1) * args.batch_size]]
                sub_data[sub_mask] = -1.
                sub_compressed = encoder(sub_data)
                sub_recon = decoder(sub_compressed)
                if j == num_batch - 1:
                    compressed[indices[j * args.batch_size: data.shape[0]]] = sub_compressed
                    recon[indices[j * args.batch_size: data.shape[0]]] = sub_recon[:, 0, :]
                else:
                    compressed[indices[j * args.batch_size: (j + 1) * args.batch_size]] = sub_compressed
                    recon[indices[j * args.batch_size: (j + 1) * args.batch_size]] = sub_recon[:, 0, :]
                del sub_data, sub_compressed, sub_recon
            data = data[:, 0]
            mask = mask[:, 0]
            loss = (mse_criterion(data, recon) * ~mask).sum() / (~mask).sum()
            mse += loss.item()
            psnrs[i] = 20. * np.log10(2.) - 10. * np.log10(loss.item())
            diff = abs(data * ~mask - recon * ~mask)
            max_diff[i] = diff.max().item() / 2.
            print(sample["name"], psnrs[i], max_diff[i])
            # print(sample["name"])

            if args.save:
                compressed = compressed.permute((0, 2, 1))  # data.shape[0], feat_size, args.latent_dim
                compressed = compressed.cpu().numpy().astype(np.float32)
                np.save(os.path.join(args.root, "vp" + args.direction, "train_compressed", sample["name"][0]), compressed)
                # recon = recon.cpu().numpy().astype(np.float32)
                # recon = np.power(10., recon)
                # recon.tofile(os.path.join(args.root, "train_recon_noweight", sample["name"][0] + ".bin"))

            del recon
            del compressed

        print("====> PSNR on raw {}"
              .format(20. * np.log10(dmax - dmin) -
                      10. * np.log10(mse / len(test_dataset))))
        print("====> max difference on raw {}"
              .format(max_diff.mean()))

if __name__ == "__main__":
  main(parse_args())