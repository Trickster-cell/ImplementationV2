from torch import nn
import snntorch as snn
from SpikingSelfAttention import SSA
import torch
from snntorch import surrogate, utils


beta = 0.9
surrogate_grad = surrogate.fast_sigmoid(slope=25)

class EfficientMultiHeadedAttention(nn.Module):
  def __init__(self, channels, reduction_ratio, num_heads):
    super().__init__()
    self.reducer = nn.Sequential(
                                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio),
                                snn.Leaky(beta= beta, init_hidden=True, spike_grad=surrogate_grad, output=True),
                                )
    
    self.att = SSA(channels, num_heads= num_heads)

  def forward(self, x):
    utils.reset(self.reducer)
    utils.reset(self.att) # aise hi kar diye.. acha lg rha tha
    batch_size, chnls, h, w = x.shape
    reduced_x = self.reducer(x)
    # attention needs tensor of shape (batch,channels,sequence_length)
    reduced_x = reduced_x[0]  ## kyo kiya... pata ni.. dekhna hai, abhi bs h ki shape me change tha
    # reduced_x = torch.stack(reduced_x, dim=0)
    # print(reduced_x.shape)
    reduced_x = reduced_x.reshape(reduced_x.size(0),reduced_x.size(1),-1)
    #print(f"reduced x shape: {reduced_x.shape}")
    x = x.reshape(x.size(0),x.size(1),-1)
    #print(f"x shape: {x.shape}")
    out= self.att(x, reduced_x, reduced_x)[0]
    #print(f"output shape: {out.shape}")
    #reshape it back to (batch, channels, height, width)
    out = out.reshape(out.size(0), out.size(1), h, w)
    #print(out.sum())
    return out
  
#x= torch.randn((1,8,64,64))
#block = EfficientMultiHeadedAttention(8,4,8)
#print(block(x).shape)