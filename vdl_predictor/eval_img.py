import os
import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.metrics.pairwise import euclidean_distances
import pyemd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

import pdb

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--root", required=True, type=str,
                        help="root of the dataset")
parser.add_argument("--tf", required=True, type=int,
                        help="transfer function index")
parser.add_argument("--mode", required=True, type=str,
                        help="fused, data_pred, idw_interp or img_pred")

def compute_emd(im1, im2, cost_mat, l_bins=8, a_bins=12, b_bins=12):
    lab_im1 = color.rgb2lab(im1.astype(np.uint8))
    lab_im1 = lab_im1.reshape((lab_im1.shape[0] * lab_im1.shape[1], lab_im1.shape[2]))
    lab_hist_1, _ = np.histogramdd(lab_im1, bins=(l_bins, a_bins, b_bins), range=[[0., 100.], [-86.185, 98.254], [-107.863, 94.482]], normed=False)

    lab_im2 = color.rgb2lab(im2.astype(np.uint8))
    lab_im2 = lab_im2.reshape((lab_im2.shape[0] * lab_im2.shape[1], lab_im2.shape[2]))
    lab_hist_2, _ = np.histogramdd(lab_im2, bins=(l_bins, a_bins, b_bins), range=[[0., 100.], [-86.185, 98.254], [-107.863, 94.482]], normed=False)

    n_bins = l_bins * a_bins * b_bins
    lab_hist_1 = lab_hist_1.reshape((n_bins))
    lab_hist_2 = lab_hist_2.reshape((n_bins))
    img_res = lab_im1.shape[0]
    lab_hist_1 /= img_res
    lab_hist_2 /= img_res
    return pyemd.emd(lab_hist_1, lab_hist_2, cost_mat)


def compute_emd_cost_mat(l_bins=8, a_bins=12, b_bins=12):
    n_bins = l_bins * a_bins * b_bins
    index_mat = np.zeros((l_bins, a_bins, b_bins, 3))
    for idx in range(l_bins):
        for jdx in range(a_bins):
            for kdx in range(b_bins):
                index_mat[idx, jdx, kdx] = np.array([idx, jdx, kdx])
    index_mat = index_mat.reshape(n_bins, 3)
    all_dists = euclidean_distances(index_mat, index_mat)
    return all_dists / np.max(all_dists)


if __name__ == '__main__':
    emd_cost_mat = compute_emd_cost_mat()

    args = parser.parse_args()
    root1 = os.path.join(args.root, "tf" + str(args.tf), "test")
    root2 = os.path.join(args.root, "tf" + str(args.tf), args.mode)

    fh = open(os.path.join(args.root, "../test", "names.txt"))
    filenames = []
    for line in fh:
        line = line.strip("\r\n")
        filenames.append(line)

    fh = open(os.path.join(args.root, "../viewpoints.txt"))
    fh.readline()
    viewpoints = []
    for line in fh:
        phi, theta = line.strip("\r\n").split()
        vp = (float(phi), float(theta))
        viewpoints.append(vp)

    num_data = len(filenames)
    num_vps = len(viewpoints)
    ssims = np.zeros((num_vps, num_data)).astype(np.float64)
    ssims_fused = np.zeros((num_vps, num_data)).astype(np.float64)
    emds = np.zeros((num_vps, num_data)).astype(np.float64)
    emds_fused = np.zeros((num_vps, num_data)).astype(np.float64)

    for i in range(num_data):
        filename = filenames[i]
        for j in range(num_vps):
            img1 = io.imread(os.path.join(root1, filename, str(j) + ".png"))
            img2 = io.imread(os.path.join(root2, filename, str(j) + ".png"))

            ssim = compute_ssim(img1, img2, data_range=255., multichannel=True)
            ssims[j, i] = ssim
            emd = compute_emd(img1, img2, emd_cost_mat)
            emds[j, i] = emd

        print("{}, SSIM: {:f}, EMD: {:f}".format(filename, ssims[:, i].mean(), emds[:, i].mean()))
    np.save(os.path.join(args.root, "res", "ssim_" + args.mode + ".npy"), ssims)
    np.save(os.path.join(args.root, "res", "emd_" + args.mode + ".npy"), emds)
    print("SSIM: {:f}, EMD: {:f}".format(ssims.mean(), emds.mean()))
