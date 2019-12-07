"""
The datastructures manipulated by the operations in operations.py
"""

from torch import nn

import collections
import numpy as np

from mlca.helpers.nn import SimpleConvNet, MLP

CHW = collections.namedtuple('CHW', ('channels', 'height', 'width'))

class CNNModule(nn.Module):
    def __init__(self, environment):
        super().__init__()

        observation_space = environment.observation_space.shape
        image_size = CHW(
            observation_space[2], 
            observation_space[0], 
            observation_space[1])

        self.conv = SimpleConvNet(image_size.channels, 1, [], [3], {"USE_BATCH_NORM": True})
        self.mlp = MLP(
            self.conv.output_size(
                (image_size.width, image_size.height)), 32, [32, 32]
        )

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x

class ObservationMLPModule(nn.Module):
    def __init__(self, environment):
        super().__init__()

        self.mlp = MLP(
            np.prod(environment.observation_space.shape),
            32, [16, 32, 64])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x

class Ensemble(nn.Module):
    def __init__(self, modules, environment):
        super().__init__()

        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        return [module(x) for module in self.module_list]
