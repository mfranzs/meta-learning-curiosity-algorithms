import torch
from torch import nn
from mlca.helpers.nn import SimpleConvNet, MLP

class Ensemble(nn.Module):
  def __init__(self, environment):
    pass

  def forward(self, x):
    x = self.conv(x)
    x = self.mlp(x)
    return x
