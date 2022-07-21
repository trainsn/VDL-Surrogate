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

from mpas import *
from aemodel import *
from generator import Generator

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
                        help="parallel direction: x, y or z")

    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size for training (default: 1)")

    parser.add_argument("--bwsa", type=float, default=2.5,
                        help="config_bulk_wind_stress_amp (BwsA) (default 2.5)")
    parser.add_argument("--kappa", type=float, default=900.0,
                        help="config_gm_constant_kappa (default 900.0)")
    parser.add_argument("--cvmix", type=float, default=0.625,
                        help="config_cvmix_kpp_criticalbulkrichardsonnumber (default 0.625)")
    parser.add_argument("--mom", type=float, default=200.0,
                        help="config_mom_del2 (default: 200.0)")
    parser.add_argument("--param-idx", required=True, type=int,
                        help="the index of the input parameter")
    parser.add_argument("--step", type=int, default=40,
                        help="the steps of selected parameter")

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

    g_model = Generator(args.direction, args.latent_dim, args.dsp, args.dspe, args.ch)
    g_model.cuda()
    decoder = Decoder(args.latent_dim, args.ae_ch)
    if args.sn:
        decoder = add_sn(decoder)
    decoder.cuda()

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint["epoch"]
        g_model.load_state_dict(checkpoint["g_model_state_dict"])
        # g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        print("=> loaded checkpoint {} (epoch {})"
                .format(args.resume, checkpoint["epoch"]))
        del checkpoint

    if args.ae_resume:
        if os.path.isfile(args.ae_resume):
            print("=> loading checkpoint {}".format(args.ae_resume))
        checkpoint = torch.load(args.ae_resume, map_location='cpu')
        args.start_epoch = checkpoint["epoch"]
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        print("=> loaded checkpoint {} (epoch {})"
              .format(args.ae_resume, checkpoint["epoch"]))
        del checkpoint

    g_model.train()  # In BatchNorm, we still want the mean and var calculated from the current instance
    decoder.eval()

    params = ["BwsA", "kappa", "cvmix", "mom"]
    BwsAMin, BwsAMax = 0.0, 5.0
    kappaMin, kappaMax = 300.0, 1500.0
    cvmixMin, cvmixMax = 0.25, 1.0
    momMin, momMax = 100.0, 300.0

    step = 2. / (args.step - 1)
    sparams_sweep = Variable(torch.zeros(1, 4), requires_grad=True)

    depth_shape = 768
    lat_shape = 768
    lon_shape = 1536
    reduce_size = 16

    grads = np.zeros(args.step)
    for i in tqdm(range(args.step)):
        if args.param_idx == 0:
            sparams_sweep[0, 0] = -1. + i * step
            sparams_sweep[0, 1] = (args.kappa - kappaMin) / (kappaMax - kappaMin)
            sparams_sweep[0, 2] = (args.cvmix - cvmixMin) / (cvmixMax - cvmixMin)
            sparams_sweep[0, 3] = (args.mom - momMin) / (momMax - momMin)
        elif args.param_idx == 1:
            sparams_sweep[0, 0] = (args.bwsa - BwsAMin) / (BwsAMax - BwsAMin)
            sparams_sweep[0, 1] = -1. + i * step
            sparams_sweep[0, 2] = (args.cvmix - cvmixMin) / (cvmixMax - cvmixMin)
            sparams_sweep[0, 3] = (args.mom - momMin) / (momMax - momMin)
        elif args.param_idx == 2:
            sparams_sweep[0, 0] = (args.bwsa - BwsAMin) / (BwsAMax - BwsAMin)
            sparams_sweep[0, 1] = (args.kappa - kappaMin) / (kappaMax - kappaMin)
            sparams_sweep[0, 2] = -1. + i * step
            sparams_sweep[0, 3] = (args.mom - momMin) / (momMax - momMin)
        elif args.param_idx == 3:
            sparams_sweep[0, 0] = (args.bwsa - BwsAMin) / (BwsAMax - BwsAMin)
            sparams_sweep[0, 1] = (args.kappa - kappaMin) / (kappaMax - kappaMin)
            sparams_sweep[0, 2] = (args.cvmix - cvmixMin) / (cvmixMax - cvmixMin)
            sparams_sweep[0, 3] = -1. + i * step
        sparams_sweep = sparams_sweep.cuda()

        fake_compressed_data = g_model(sparams_sweep)[0]
        fake_compressed_data = fake_compressed_data.permute((1, 2, 0, 3))
        if args.direction == "lon":
            fake_compressed_data = fake_compressed_data.view((-1, args.latent_dim, lon_shape // reduce_size))
            num_batch = (lat_shape * depth_shape // 4 - 1) // args.batch_size + 1
            indices = np.arange(lat_shape * depth_shape // 4)
        elif args.direction == "lat":
            fake_compressed_data = fake_compressed_data.view((-1, args.latent_dim, lat_shape // reduce_size))
            num_batch = (lon_shape * depth_shape // 4 - 1) // args.batch_size + 1
            indices = np.arange(lon_shape * depth_shape // 4)
        elif args.direction == "depth":
            fake_compressed_data = fake_compressed_data.view((-1, args.latent_dim, depth_shape // reduce_size))
            num_batch = (lon_shape * lat_shape // 4 - 1) // args.batch_size + 1
            indices = np.arange(lon_shape * lat_shape // 4)

        for j in range(num_batch):
            if j == num_batch - 1:
                sub_compressed = fake_compressed_data[indices[j * args.batch_size: indices.shape[0]]]
            else:
                sub_compressed = fake_compressed_data[indices[j * args.batch_size: (j + 1) * args.batch_size]]
            sub_recon = decoder(sub_compressed).squeeze(1)
            grad = torch.autograd.grad(sub_recon.norm(p=1), sparams_sweep, retain_graph=True)
            grads[i] += abs(grad[0][0, args.param_idx].item())
            del sub_compressed, sub_recon
            # print("finish calculating the {:d}th batch".format(j))
        del fake_compressed_data
        grads[i] /= num_batch
        # print("finish calculating the {:d}th data".format(i))

    np.save(os.path.join(args.root, "grads_" + params[args.param_idx] + ".npy"), grads)

if __name__ == "__main__":
    main(parse_args())