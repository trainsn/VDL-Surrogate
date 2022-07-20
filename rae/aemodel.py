# architecture
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from aeresblock import BasicBlockEncoder, FirstBlockEncoder, BasicBlockDecoder

class Encoder(nn.Module):
  def __init__(self, latent_dim=3, ch=64):
    super(Encoder, self).__init__()
    self.latent_dim = latent_dim
    self.ch = ch

    self.BE0 = FirstBlockEncoder(1, ch, kernel_size=3, stride=1, padding=1)
    self.BE1 = BasicBlockEncoder(ch, ch * 2, kernel_size=3, stride=1, padding=1)
    self.BE2 = BasicBlockEncoder(ch * 2, ch * 4, kernel_size=3, stride=1, padding=1)
    self.BE3 = BasicBlockEncoder(ch * 4, ch * 8, kernel_size=3, stride=1, padding=1)
    self.BE4 = BasicBlockEncoder(ch * 8, latent_dim, kernel_size=3, stride=1, padding=1, downsample=False)
    self.tanh = nn.Tanh()

  def forward(self, input):
    x = self.BE0(input)
    del input
    x = self.BE1(x)
    x = self.BE2(x)
    x = self.BE3(x)
    x = self.BE4(x)
    x = self.tanh(x)
    return x

class Decoder(nn.Module):
  def __init__(self, latent_dim=3, ch=64):
    super(Decoder, self).__init__()
    self.latent_dim = latent_dim
    self.ch = ch

    self.BD0 = BasicBlockDecoder(latent_dim, ch * 8, kernel_size=3, stride=1, padding=1, upsample=False)
    self.BD1 = BasicBlockDecoder(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1)
    self.BD2 = BasicBlockDecoder(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1)
    self.BD3 = BasicBlockDecoder(ch * 2, ch, kernel_size=3, stride=1, padding=1)
    self.BD4 = BasicBlockDecoder(ch, 1, kernel_size=3, stride=1, padding=1)
    self.tanh = nn.Tanh()

  def forward(self, input):
    x = self.BD0(input)
    del input
    x = self.BD1(x)
    x = self.BD2(x)
    x = self.BD3(x)
    x = self.BD4(x)
    x = self.tanh(x)
    return x

class AutoEncoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(AutoEncoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input):
    x = self.encoder(input)
    del input
    x = self.decoder(x)
    return x



