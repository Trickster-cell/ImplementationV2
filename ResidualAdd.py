import torch
from torch import nn
import numpy as np



class ResidualAdd(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, **kwargs):
    out = self.fn(x, **kwargs)
    x=x+out
    return x