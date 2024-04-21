import torch
from torch import nn
import numpy as np

import snntorch as snn
from snntorch import surrogate, utils


beta = 0.9
surrogate_grad = surrogate.fast_sigmoid(slope=25)

#mix-ffn layer
class Mix_FFN(nn.Module):
  '''
  dense layer followed by convolution layer with gelu activation then another
  dense layer
  '''
  def __init__(self, channels, expansion=4):
    super().__init__()
    self.channels = channels
    self.expansion = expansion
    self.dense1 = nn.Conv2d(channels, channels, kernel_size=1)
    self.lif1 = snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad)

    self.conv = nn.Conv2d(channels,
                              channels*expansion,
                              kernel_size=3,
                              groups= channels,
                              padding=1)
    self.lif2 = snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad)
    
    # self.gelu= nn.GELU()
    self.dense2 = nn.Conv2d(channels*expansion, channels, kernel_size=1)
    self.lif3 = snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad, output=True)

  def forward(self,x):
    # utils.reset(self.dense1)
    utils.reset(self.lif1)
    # utils.reset(self.conv)
    utils.reset(self.lif2)
    # utils.reset(self.gelu)
    # utils.reset(self.dense2)
    utils.reset(self.lif3)
    x=self.dense1(x)
    x=self.lif1(x)
    x=self.conv(x)
    x=self.lif2(x)
    # x=self.gelu(x)
    x=self.dense2(x)
    x=self.lif3(x)
    x=x[0]
    # print(x.shape)
    return x
