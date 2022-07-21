# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockGenerator

import pdb

class Generator(nn.Module):
  def __init__(self, latent_dim, dsp=3, dspe=512, ch=64):
    # dsp  - dimensions of the simulation parameters
    # dvp  - dimensions of the visualization parameters
    # dspe - dimensions of the simulation parameters' encode
    # dvpe - dimensions of the visualization parameters' encode
    # ch   - channel multiplier
    super(Generator, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.ch = ch

    # parameters subnet
    self.params_subnet = nn.Sequential(
      nn.Linear(dsp, dspe // 2), nn.ReLU(),
      nn.Linear(dspe // 2, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU(),
      nn.Linear(dspe, ch * 16 * 6 * 6 * 4, bias=True)
    ).cuda(0)

    # image generation subnet
    self.BG0 = BasicBlockGenerator(ch * 16, ch * 16, [12, 12, 4], kernel_size=3, stride=1, padding=1).cuda(0)
    self.BG1 = BasicBlockGenerator(ch * 16, ch * 8, [24, 24, 8], kernel_size=3, stride=1, padding=1).cuda(0)
    self.BG2 = BasicBlockGenerator(ch * 8, ch * 8, [48, 48, 8], kernel_size=3, stride=1, padding=1).cuda(0)
    self.BG3 = BasicBlockGenerator(ch * 8, ch * 4, [96, 96, 16], kernel_size=3, stride=1, padding=1).cuda(0)
    self.BG4 = BasicBlockGenerator(ch * 4, ch * 2, [192, 192, 16], kernel_size=3, stride=1, padding=1).cuda(0)
    self.BG5 = BasicBlockGenerator(ch * 2, ch, [384, 384, 32], kernel_size=3, stride=1, padding=1).cuda(0)
    self.BG6 = nn.Sequential(
      nn.BatchNorm3d(ch),
      nn.ReLU(),
      nn.Conv3d(ch, latent_dim, kernel_size=3, stride=1, padding=1),
    ).cuda(0)
    self.tanh = nn.Tanh().cuda(0)

  def forward(self, sp):
    sp = self.params_subnet(sp)
    x = sp.view(sp.size(0), self.ch * 16, 6, 6, 4)
    x = self.BG0(x)
    x = self.BG1(x)
    x = self.BG2(x)
    x = self.BG3(x)
    x = self.BG4(x)
    x = self.BG5(x)
    x = self.BG6(x)
    x = self.tanh(x)

    return x
