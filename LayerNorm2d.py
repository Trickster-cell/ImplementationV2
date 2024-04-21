import torch
from torch import nn
import numpy as np

# %%
#you can create a Layernorm function or class here in future
class LayerNorm2d(nn.LayerNorm):
  def forward(self,x):
    x = x.permute(0, 2, 3, 1)
    x = super().forward(x)
    x = x.permute(0, 3, 1, 2)

    return x