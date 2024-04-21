import torch
from torch import nn
import numpy as np
import snntorch as snn
from snntorch import surrogate


beta = 0.9
surrogate_grad = surrogate.fast_sigmoid(slope=25)

class DecoderBlock(nn.Sequential):
  def __init__(self, in_channels, out_channels, scale_factor: int =2):
    super().__init__(
    nn.UpsamplingBilinear2d(scale_factor = scale_factor),
    nn.Conv2d(in_channels, out_channels, kernel_size = 1),
    snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad),

    )
