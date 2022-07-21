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

from nyx import *
from aemodel import *
from generator import Generator


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Model")

    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
    parser.add_argument("--resume", type=str, default="",
                        help="path to the latest checkpoint (default: none)")
    parser.add_argument("--ae-resume", required=True, type=str,
                        help="path to AutoEncoder")

    parser.add_argument("--dsp", type=int, default=3,
                        help="dimensions of the simulation parameters (default: 3)")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--latent-dim", type=int, default=3,
                        help="latent vector dimension(default: 3)")
    parser.add_argument("--data-size", type=int, default=800,
                        help="volume data (default: 800)")
    parser.add_argument("--reduce-size", type=int, default=16,
                        help="ray reduce size (default: 16)")
    parser.add_argument("--img-size", type=int, default=512,
                        help="volume data (default: 512)")
    parser.add_argument("--ch", type=int, default=64,
                        help="channel multiplier (default: 64)")
    parser.add_argument("--ae-ch", type=int, default=64,
                        help="channel multiplier of AutoEncoder (default: 64)")
    parser.add_argument("--direction", type=str, required=True,
                        help="parallel direction: x, y or z")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")

    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")

    parser.add_argument("--omm", type=float, default=0.1375,
                        help="the total matter density (OmM) (default 0.1375)")
    parser.add_argument("--omb", type=float, default=0.0225,
                        help="the total density of baryons (OmM) (default 0.0225)")
    parser.add_argument("--h", type=float, default=0.7,
                        help="the Hubble constant (OmB) (default 0.7)")

    return parser.parse_args()


# the main function
def main(args):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m

    g_model = Generator(args.latent_dim, args.dsp, args.dspe, args.ch)
    if args.sn:
        g_model = add_sn(g_model)
    g_model.cuda()
    decoder = Decoder(args.latent_dim, args.ae_ch)
    if args.sn:
        decoder = add_sn(decoder)
    decoder.cuda()

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device("cuda:0"))
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

    OmMMin, OmMMax = 0.12, 0.155
    OmBMin, OmBMax = 0.0215, 0.0235
    hMin, hMax = 0.55, 0.85

    dmin = 8.6
    dmax = 13.6

    decoder.eval()
    g_model.train()  # In BatchNorm, we still want the mean and var calculated from the current instance

    with torch.no_grad():
        sparams = torch.zeros(1, 3).cuda()
        sparams[0, 0] = ((args.omm - OmMMin) / (OmMMax - OmMMin)) * 2. - 1.
        sparams[0, 1] = ((args.omb - OmBMin) / (OmBMax - OmBMin)) * 2. - 1.
        sparams[0, 2] = ((args.h - hMin) / (hMax - hMin)) * 2. - 1.

        fake = g_model(sparams)

        fake_compressed_data = fake[0]
        fake_compressed_data = fake_compressed_data.permute((1, 2, 0, 3))
        fake_compressed_data = fake_compressed_data.view((args.img_size * args.img_size, args.latent_dim, -1))
        fake_data = -1. * torch.ones((args.img_size * args.img_size, args.data_size)).cuda(0)
        num_batch = (args.img_size * args.img_size - 1) // args.batch_size + 1
        indices = np.arange(args.img_size * args.img_size)
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
        fake_data = fake_data.view((args.img_size, args.img_size, args.data_size))
        if args.direction == "y":
            fake_data = fake_data.permute((0, 2, 1))
        elif args.direction == "z":
            fake_data = fake_data.permute((2, 0, 1))

        output = torch.pow(10., fake_data)
        output = output.cpu().numpy()
        output.tofile(os.path.join(args.root, "case", "h{:.2f}.bin".format(args.h)))
        del output

if __name__ == "__main__":
    main(parse_args())
