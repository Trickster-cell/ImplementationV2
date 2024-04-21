import torch
from torch import nn
import numpy as np
from torchvision.ops import StochasticDepth
from LayerNorm2d import LayerNorm2d
from ResidualAdd import ResidualAdd
from MixFFN import Mix_FFN
from EfficientMultiHeadedAttention import EfficientMultiHeadedAttention


class SegFormerEncoderBlock(nn.Sequential):
  def __init__(
      self,
      channels: int,
      reduction_ratio: int = 1,
      num_heads: int = 8,
      mlp_expansion: int = 4,
      drop_path_prob: float = 0.0,
  ):

    super().__init__(
        ResidualAdd(
            nn.Sequential(
                # LayerNorm2d(channels),
                EfficientMultiHeadedAttention(channels, reduction_ratio, num_heads),
            )
        ),
        ResidualAdd(
            nn.Sequential(
                # LayerNorm2d(channels),
                Mix_FFN(channels, expansion = mlp_expansion),
                StochasticDepth(p=drop_path_prob, mode="batch")
            )
        ),
    )
