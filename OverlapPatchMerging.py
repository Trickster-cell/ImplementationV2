import torch
from torch import nn
import numpy as np
from LayerNorm2d import LayerNorm2d
import snntorch as snn
from snntorch import surrogate


beta = 0.9
surrogate_grad = surrogate.fast_sigmoid(slope=25)


class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),                    
            snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad),

            # LayerNorm2d(out_channels)
        )

