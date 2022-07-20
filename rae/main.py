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

import pdb

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")

    parser.add_argument("--data-size", type=int, default=800,
                        help="volume data (default: 800)")
    parser.add_argument("--img-size", type=int, default=512,
                        help="volume data (default: 512)")
    parser.add_argument("--latent-dim", type=int, default=3,
                        help="latent vector dimension(default: 3)")
    parser.add_argument("--ch", type=int, default=64,
                        help="channel multiplier (default: 64)")
    parser.add_argument("--direction", type=str, required=True,
                        help="parallel direction: x, y or z")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate (default: 5e-5)")
    parser.add_argument("--beta1", type=float, default=0.0,
                        help="beta1 of Adam (default: 0.0)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam (default: 0.999)")
    parser.add_argument("--load-batch", type=int, default=8,
                        help="batch size for loading (default: 8)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="use weighted L1 Loss")
    parser.add_argument("--opt-level", default='O2',
                        help='amp opt_level, default="O2"')
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number (default: 0)")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of epochs to train (default: 10000)")

    parser.add_argument("--log-every", type=int, default=10,
                        help="log training status every given number of batches (default: 10)")
    parser.add_argument("--check-every", type=int, default=20,
                        help="save checkpoint every given number of epochs (default: 20)")

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
        data_size = args.data_size,
        img_size = args.img_size,
        direction=args.direction,
        train=True,
        transform=transforms.Compose([Normalize(), ToTensor()]))

    kwargs = {"num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_dataset, batch_size=args.load_batch,
                              shuffle=False, **kwargs)

    # model
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
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

    encoder = Encoder(args.latent_dim, args.ch)
    encoder.apply(weights_init)
    if args.sn:
        encoder = add_sn(encoder)
    encoder.cuda()
    decoder = Decoder(args.latent_dim, args.ch)
    decoder.apply(weights_init)
    if args.sn:
        decoder = add_sn(decoder)
    decoder.cuda()

    AE = AutoEncoder(encoder, decoder)

    if args.weighted:
        l1_criterion = nn.L1Loss(reduction='none').cuda()
    else:
        l1_criterion = nn.L1Loss().cuda()
    train_losses, test_losses = [], []

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

    optimizer = optim.Adam(AE.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2))
    AE, optimizer = amp.initialize(AE, optimizer, opt_level=args.opt_level)

    num_bins = 10

    # main loop
    for epoch in range(args.start_epoch, args.epochs):
        AE.train()
        cursor_batch = 0
        for i, sample in enumerate(train_loader):
            data = sample["data"]
            data = data.reshape((-1, data.shape[-1])).cuda(non_blocking=True)
            data = data.unsqueeze(1)
            indices = torch.randperm(data.shape[0])
            if args.weighted:
                freq = torch.histc(data, bins=10, min=-1., max=1.)
                importance = 1. / freq / num_bins

            num_batch = (indices.shape[0] - 1) // args.batch_size + 1
            train_loss = 0.
            for j in range(num_batch):
                if j == num_batch - 1:
                    sub_data = data[indices[j * args.batch_size: data.shape[0]]]
                else:
                    sub_data = data[indices[j * args.batch_size: (j+1) * args.batch_size]]
                optimizer.zero_grad()
                fake_data = AE(sub_data)
                loss = l1_criterion(sub_data, fake_data)
                if args.weighted:
                    importance_idx = ((sub_data + 1.) / 2. * num_bins).type(torch.long)
                    sub_importance = importance[importance_idx]
                    loss = (loss * sub_importance).sum()

                with amp.scale_loss(loss, optimizer, loss_id=0) as loss_scaled:
                    loss_scaled.backward()
                optimizer.step()

                train_loss += loss.detach().item()

                cursor_batch += 1
                if cursor_batch % args.log_every == 0:
                    if args.weighted:
                        loss_print = loss.detach().item() * num_batch
                    else:
                        loss_print = loss.detach().item()
                    print("Train Epoch: {} [{}]\tL1_Loss: {:.6f}".format(
                        epoch, cursor_batch, loss_print))
                del loss

            if args.weighted:
                train_loss_print = train_loss
            else:
                train_loss_print = train_loss  / num_batch
            print("====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss_print))

        # saving...
        if (epoch + 1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save({"epoch": epoch + 1,
                        "encoder_state_dict": encoder.state_dict(),
                        "decoder_state_dict": decoder.state_dict(),
                        "train_losses": train_losses,
                        "test_losses": test_losses},
                       os.path.join(args.root, "autoencoder_" + str(epoch + 1) + ".pth.tar"))

if __name__ == "__main__":
  main(parse_args())
