import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate,  utils


beta = 0.9
surrogate_grad = surrogate.fast_sigmoid(slope=25)

class SegmentationHead(nn.Module):
  def __init__(self, channels, num_classes, num_features = 4):
    super().__init__()
    self.channels = channels
    self.num_classes = num_classes
    self.num_features = num_features

    self.dense1 = nn.Conv2d(channels*num_features, channels, kernel_size=1, bias=False)
    self.lif1 = snn.Leaky(beta= beta, spike_grad=surrogate_grad, init_hidden=True)

    #self.relu = nn.ReLU()
    #self.bn = nn.BatchNorm2d(channels)

    self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)
    self.upscale = nn.UpsamplingBilinear2d(scale_factor=4)
    self.lif2 = snn.Leaky(beta= beta, spike_grad=surrogate_grad, init_hidden=True)


  def forward(self,x):
    utils.reset(self.lif1)
    utils.reset(self.lif2)
    x=torch.cat(x, dim=1)
    x=self.dense1(x)
    x = self.lif1(x)
    # x=self.relu(x)
    # x=self.bn(x)
    x=self.predict(x)
    x=self.upscale(x)
    x=self.lif2(x)

    return x
