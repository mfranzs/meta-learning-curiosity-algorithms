import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mlca.helpers.config import DefaultDevice

class SimpleConvNet(nn.Module):
  def __init__(self, input_channel, output_channel, hidden_layer_channels, kernel_sizes, module_params, downscale=True):
    super().__init__()
    self.layers = nn.ModuleList()
    self.downscale = downscale
    self.module_params = module_params
    self.batch_norms = nn.ModuleList()
    self.kernel_sizes = kernel_sizes

    input_channels = [input_channel] + hidden_layer_channels
    output_channels = hidden_layer_channels + [output_channel]

    assert(len(kernel_sizes) == len(hidden_layer_channels) + 1)

    for i in range(len(kernel_sizes)):
      padding = 0 if self.downscale else int((kernel_sizes[i] - 1) / 2)
      self.layers.append(
          nn.Conv2d(input_channels[i], output_channels[i], kernel_size=kernel_sizes[i], padding=padding))
      if self.module_params.get("USE_BATCH_NORM", False):
        self.batch_norms.append(nn.BatchNorm2d(output_channels[i]))

  def output_size(self, input_shape):
    if self.downscale == False:
      return input_shape[0] * input_shape[1]
    else: 
      size = 1

      for axis_size in input_shape:
        original_axis_size = axis_size
        for kernel_size in self.kernel_sizes:
          stride = 1
          axis_size = (axis_size - (kernel_size - 1) - 1) / stride + 1
          assert int(axis_size) == axis_size

        assert axis_size > 0, "The kernels of your SimpleConvNet shrink the input to size 0. Problematic axis length: " + str(original_axis_size)
        size *= axis_size
      
      return int(size)

  def forward(self, x):
    for i in range(len(self.layers)):
      layer = self.layers[i]
      if i == len(self.layers) - 1:
        x = layer(x)
      else:
        if self.module_params.get("USE_BATCH_NORM", False):
            bn = self.batch_norms[i]
            x = F.relu(bn(layer(x)))
        else:
            x = F.relu(layer(x))

    return x

class MLP(nn.Module):
  def __init__(self, input_size, output_size, hiddel_layer_sizes):
    super().__init__()
    self.layers = nn.ModuleList()

    input_sizes = [input_size] + list(hiddel_layer_sizes)
    output_sizes = list(hiddel_layer_sizes) + [output_size]

    for i in range(len(input_sizes)):
      self.layers.append(
          nn.Linear(input_sizes[i], output_sizes[i]))

  def forward(self, x):
    for i in range(len(self.layers)):
      layer = self.layers[i]
      if i == len(self.layers) - 1:
        x = layer(x)
      else:
        x = F.relu(layer(x))

    return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(1, -1)

def one_hot(labels, num_classes):
    """Embedding labels to one-hot form.
    Source: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes, device=DefaultDevice.current())
    return y[labels]
