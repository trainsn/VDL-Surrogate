# Residual block architecture

import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlockGenerator(nn.Module):
  def __init__(self, in_channels, out_channels, out_size, kernel_size=3, stride=1,
               padding=1, activation=F.relu, upsample=True):
    super(BasicBlockGenerator, self).__init__()

    self.out_size = out_size
    self.activation = activation
    self.upsample = upsample
    self.conv_res = None
    if self.upsample or in_channels != out_channels:
      self.conv_res = nn.Conv3d(in_channels, out_channels,
                                1, 1, 0, bias=False)

    self.bn0 = nn.BatchNorm3d(in_channels)
    self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=False)

    self.bn1 = nn.BatchNorm3d(out_channels)
    self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=False)

  def forward(self, x):
    residual = x
    if self.upsample:
      residual = F.interpolate(residual, size=self.out_size)
    if self.conv_res is not None:
      residual = self.conv_res(residual)

    out = self.bn0(x)
    out = self.activation(out)
    if self.upsample:
      out = F.interpolate(out, size=self.out_size)
    out = self.conv0(out)

    out = self.bn1(out)
    out = self.activation(out)
    out = self.conv1(out)

    return out + residual
