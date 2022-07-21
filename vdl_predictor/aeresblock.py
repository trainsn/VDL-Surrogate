# Residual block architecture

import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlockEncoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu, downsample=True):
    super(BasicBlockEncoder, self).__init__()

    self.activation = activation
    self.downsample = downsample
    self.conv_res = None
    if self.downsample or in_channels != out_channels:
      self.conv_res = nn.Conv1d(in_channels, out_channels,
                                1, 1, 0, bias=True, padding_mode="circular")

    self.in0 = nn.InstanceNorm1d(in_channels)
    self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=True, padding_mode="circular")
    self.in1 = nn.InstanceNorm1d(in_channels)
    self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=True, padding_mode="circular")

  def forward(self, x):
    residual = x
    if self.conv_res is not None:
      residual = self.conv_res(residual)
    if self.downsample:
      residual = F.avg_pool1d(residual, kernel_size=2)

    out = self.conv0(x)
    out = self.in0(out)
    out = self.activation(out)

    out = self.conv1(out)
    out = self.in1(out)
    out = self.activation(out)

    if self.downsample:
      out = F.avg_pool1d(out, kernel_size=2)

    return out + residual

class FirstBlockEncoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu):
    super(FirstBlockEncoder, self).__init__()

    self.activation = activation

    self.conv_res = nn.Conv1d(in_channels, out_channels,
                              1, 1, 0, bias=True, padding_mode="circular")
    self.in0 = nn.InstanceNorm1d(in_channels)
    self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=True, padding_mode="circular")

    self.in1 = nn.InstanceNorm1d(in_channels)
    self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=True, padding_mode="circular")

  def forward(self, x):
    residual = self.conv_res(x)
    residual = F.avg_pool1d(residual, kernel_size=2)

    out = self.conv0(x)
    out = self.in0(out)
    out = self.activation(out)
    out = self.conv1(out)
    out = self.in1(out)

    out = F.avg_pool1d(out, kernel_size=2)

    return out + residual

class BasicBlockDecoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu, upsample=True):
    super(BasicBlockDecoder, self).__init__()

    self.activation = activation
    self.upsample = upsample
    self.conv_res = None
    if self.upsample or in_channels != out_channels:
      self.conv_res = nn.Conv1d(in_channels, out_channels,
                                1, 1, 0, bias=False, padding_mode="circular")

    self.in0 = nn.InstanceNorm1d(in_channels)
    self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=False, padding_mode="circular")

    self.in1 = nn.InstanceNorm1d(out_channels)
    self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=False, padding_mode="circular")

  def forward(self, x):
    residual = x
    if self.upsample:
      residual = F.interpolate(residual, scale_factor=2)
    if self.conv_res is not None:
      residual = self.conv_res(residual)

    out = self.in0(x)
    out = self.activation(out)
    if self.upsample:
      out = F.interpolate(out, scale_factor=2)
    out = self.conv0(out)

    out = self.in1(out)
    out = self.activation(out)
    out = self.conv1(out)

    return out + residual
