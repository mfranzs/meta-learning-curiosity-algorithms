"""
The program types that variables in our synthesized programs can have.
"""

import torch
from torch import nn
import torch.optim as optim
import typing
import numpy as np
from typing import Any, Dict
import typing

from mlca.helpers.config import DefaultDevice
from mlca.helpers.nn import MLP
from mlca.helpers.torch_knn import TorchKNN, TorchKNNRegressor
import mlca.helpers.statistics.welfords_std
from mlca.VAE_CVAE_MNIST_mod.models import VAE

import mlca.datastructures as datastructures
from mlca.test_synthesized_programs_experiments import TspParams

# ================
# Types
# ================

class Type:
  must_be_constant = False
  DEPRECATED = False

  def __init__(self):
    self.has_gradients = False
    self.might_have_bounded_minimum = False
    self.is_constant = False  

  @classmethod
  def is_valid_value(cls, value: Any):
    return True

  @classmethod
  def is_correctly_formatted_value(cls, value: Any):
    return True

# ================
# Supertype Machinery
# ================

supertypes: Dict[Type, Type] = {}
subtypes: Dict[Type, typing.List[Type]] = {}

def register_supertype(supertype: Any):
  def _register_supertype(program_type: Type):
    assert program_type not in supertypes, f"{program_type} already has a supertype!"
    supertypes[program_type] = supertype

    if supertype not in subtypes:
      subtypes[supertype] = []
    subtypes[supertype].append(program_type)

    return program_type
  return _register_supertype

def all_supertypes(program_type: Type):
  if program_type not in supertypes:
    return [Type]
  else:
    supertype = supertypes[program_type]
    return [supertype] + all_supertypes(supertype)

def type_and_supertypes(program_type: Type):
  return [program_type] + all_supertypes(program_type)

def all_subtypes(supertype: Type):
  return subtypes.get(supertype, [])

def equal_or_supertype(program_type: Type, potential_supertype: Type):
  return potential_supertype == Type \
    or program_type == potential_supertype \
      or potential_supertype in all_supertypes(program_type)


# ==================
# Numbers
# ==================
class RealNumber(Type):
  value_class = torch.Tensor
  short_name = "ℕ"
  @classmethod
  def is_valid_value(cls, value):
    return not torch.isnan(value).byte().any()

@register_supertype(RealNumber)
class BinaryNumber(Type):
  short_name = "ℕ"
  value_class = torch.LongTensor

@register_supertype(RealNumber)
class NonNegativeNumber(Type):
  short_name = "ℕ"
  value_class = torch.Tensor

@register_supertype(RealNumber)
class Integer(Type):
  short_name = "ℕ"
  value_class = torch.Tensor

  @staticmethod
  def matches_type(i):
    return type(i) == 'number' and isinstance(i, int)

# ==================
# Feature Vectors
# ==================

class Tensor(Type):
  value_class = torch.Tensor
  tensor_value_shape = None

  MAX_ALLOWED_TENSOR_VALUE = 1e4 
  MIN_ALLOWED_TENSOR_VALUE = - 1e4

  @classmethod
  def is_valid_value(cls, value):
    return not torch.isnan(value).any() \
      and not value.max() > cls.MAX_ALLOWED_TENSOR_VALUE \
      and not value.min() < cls.MIN_ALLOWED_TENSOR_VALUE \

  @classmethod
  def is_correctly_formatted_value(cls, value):
    return not (cls.tensor_value_shape is not None and value.shape[1:] != cls.tensor_value_shape)

class FeatureVector(Tensor):
  tensor_value_shape: Any

@register_supertype(FeatureVector)
class FeatureVector32(FeatureVector):
  short_name = "𝔽"
  value_class = torch.Tensor
  tensor_value_shape = (32, )

@register_supertype(FeatureVector)
class FeatureVectorActionSpace(FeatureVector):
  value_class = torch.Tensor

@register_supertype(FeatureVector)
class FeatureVector1(FeatureVector):
  value_class = torch.Tensor
  tensor_value_shape = (1, )

# ==================
# Misc.
# ==================

class ImageTensor(Tensor):
  value_class = torch.Tensor

class Action(Type):
  value_class = int
  DEPRECATED = True

class Void(Type):
  value_class = type(None)

# ==================
# Lists
# ==================
class List(Type):
  value_class = list

@register_supertype(List)
class ListFeatureVector(Type):
  pass

@register_supertype(ListFeatureVector)
class ListFeatureVector32(Type):
  short_name = "[𝔽]"
  list_contents_type = FeatureVector32
  value_class = list


@register_supertype(ListFeatureVector)
class ListFeatureVectorActionSpace(Type):
  list_contents_type = FeatureVectorActionSpace
  value_class = list

@register_supertype(List)
class ListImageTensor(Type):
  list_contents_type = ImageTensor
  value_class = list

@register_supertype(List)
class ListRealNumber(Type):
  list_contents_type = RealNumber
  value_class = list

# ==================
# Optimizers
# ==================
class Optimizer(Type):
  def create_empty(self, environment, data_structure_values):
    raise NotImplementedError()

@register_supertype(Optimizer)
class AdamOptimizer(Type):
  def create_empty(self, environment, data_structure_values):
    torch_data_structures = [
      d for d in data_structure_values.values() if isinstance(d, nn.Module)
    ]
    nn_modules = nn.ModuleList(torch_data_structures)
    return optim.Adam(nn_modules.parameters(), lr=TspParams.current().LEARNING_RATE)

# ==================
# Data structures
# ==================

class DataStructure(Type):
  def create_empty(self, environment, policy):
    raise NotImplementedError()

class Counter(DataStructure):
  pass

@register_supertype(RealNumber)
class Constant(DataStructure):
  value_class = torch.Tensor
  must_be_constant = True

  def __init__(self, constant_value):
    super().__init__()
    self.constant_value = constant_value

  def create_empty(self, environment, policy):
    if TspParams.current().REAL_BATCH_REWARD_COMPUTATION:
      if TspParams.current().SHARE_CURIOSITY_MODULE_IN_TRIAL:
        size = TspParams.current().STEPS_PER_CURIOSITY_UPDATE
        #TspParams.current().PPO_FRAMES_PER_PROC"]) * params["NUM_ROLLOUTS_PER_TRIAL
      else:
        size = TspParams.current().STEPS_PER_CURIOSITY_UPDATE
        #", TspParams.current().PPO_FRAMES_PER_PROC)
    else:
      size = 1
    return torch.ones(size, device=DefaultDevice.current()) * self.constant_value

class NeuralNetworkWeights(DataStructure):
  pass

@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsConditionalVAE(DataStructure):
  short_name = "C-VAE Weights "
  def create_empty(self, environment, policy):
    return VAE(
      [32, 16],
      4,
      [16, 16], 
      conditional=True
    )


@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsObservationToFeatureVector32(DataStructure):
  short_name = "Weights Obs → 32"
  def create_empty(self, environment, policy):
    if len(environment.observation_space.shape) == 3:
      return datastructures.CNNModule(environment).to(DefaultDevice.current())
    else:
      return datastructures.ObservationMLPModule(environment).to(DefaultDevice.current())

@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsFeatureVector64ToFeatureVector32(DataStructure):
  short_name = "Weights 64 → 32"
  def create_empty(self, environment, policy):
    return MLP(64, 32, [32, 32]).to(DefaultDevice.current())


@register_supertype(NeuralNetworkWeights)
class NearestNeighborSmall(DataStructure):
  def create_empty(self, environment, policy):
    return TorchKNN(TspParams.current().KNN_BUFFER_SIZE_SMALL, 32, 5)


@register_supertype(NeuralNetworkWeights)
class NearestNeighborLarge(DataStructure):
  def create_empty(self, environment, policy):
    return TorchKNN(TspParams.current().KNN_BUFFER_SIZE_LARGE, 32, 5)
# Backwards compatibility:
NearestNeighbor = NearestNeighborLarge

@register_supertype(NeuralNetworkWeights)
class NearestNeighborRegressor(DataStructure):
  def create_empty(self, environment, policy):
    return TorchKNNRegressor(TspParams.current().KNN_BUFFER_SIZE_REGRESSOR, 32, 5)

@register_supertype(NeuralNetworkWeights)
class LSTM32(DataStructure):
  class LSTM32Store(nn.Module):
    def __init__(self):
      super().__init__()

      self.lstm = torch.nn.LSTM(
        input_size = 32, 
        hidden_size = 32, 
      )

      self.cur_h = np.zeros(1 * 1, 1, 32) # (num_layers * num_directions, batch, hidden_size)
      self.cur_c = np.zeros(1 * 1, 1, 32) # (num_layers * num_directions, batch, hidden_size)

    def forward(self, x):
      output, hc = self.lstm(x, (self.cur_h, self.cur_c))
      h, c = hc
      self.cur_h = h
      self.cur_c = c
      return output

  def create_empty(self, environment, policy):
    return LSTM32Store()

@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(DataStructure):
  short_name = "Weights 64 → # Actions"
  def create_empty(self, environment, policy):
    return MLP(64, get_action_space_size(environment.action_space), [32, 32]).to(DefaultDevice.current())

@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace(DataStructure):
  short_name = "Weights 32 → # Actions"
  def create_empty(self, environment, policy):
    return MLP(32, get_action_space_size(environment.action_space), [32, 32]).to(DefaultDevice.current())

@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(DataStructure):
  short_name = "Weights # Actions → 32"
  def create_empty(self, environment, policy):
    return MLP(get_action_space_size(environment.action_space), 32, [32, 32]).to(
      DefaultDevice.current())


@register_supertype(NeuralNetworkWeights)
class NeuralNetworkWeightsFeatureVector32ToFeatureVector32(DataStructure):
  short_name = "Weights 32 → 32"

  def create_empty(self, environment, policy):
    return MLP(32, 32, [32, 32]).to(DefaultDevice.current())

@register_supertype(NeuralNetworkWeights)
class EnsembleWeightsImageTo32(DataStructure):
  short_name = "Weights Image → 32x5"
  NUM_MODELS = 5

  def _make_network(self, environment):
    if len(environment.observation_space.shape) == 3:
      return datastructures.CNNModule(environment).to(DefaultDevice.current())
    else:
      return datastructures.ObservationMLPModule(environment).to(DefaultDevice.current())

  def create_empty(self, environment, policy):
    return datastructures.Ensemble(
        [self._make_network(environment) for i in range(self.NUM_MODELS)],
      environment
    ).to(DefaultDevice.current())

@register_supertype(NeuralNetworkWeights)
class EnsembleWeights32To32(DataStructure):
  short_name = "Weights 32 → 32x5"
  NUM_MODELS = 5

  def create_empty(self, environment, policy):
    return datastructures.Ensemble(
      [MLP(32, 32, [32, 32]) for i in range(self.NUM_MODELS)],
      environment
    ).to(DefaultDevice.current())

@register_supertype(NeuralNetworkWeights)
class EnsembleWeightsTwo32To32(DataStructure):
  short_name = "Weights 64 → 32x5"
  NUM_MODELS = 5

  def create_empty(self, environment, policy):
    return datastructures.Ensemble(
        [MLP(32 * 2, 32, [32, 32]) for i in range(self.NUM_MODELS)],
        environment
    ).to(DefaultDevice.current())


@register_supertype(NeuralNetworkWeights)
class EnsembleWeights32AndActionTo32(DataStructure):
  short_name = "Weights 32 + #Actions → 32x5"
  NUM_MODELS = 5

  def create_empty(self, environment, policy):
    return datastructures.Ensemble(
        [MLP(32 + get_action_space_size(environment.action_space), 32, [32, 32]) 
          for i in range(self.NUM_MODELS)],
        environment
    ).to(DefaultDevice.current())

@register_supertype(DataStructure)
class Policy(DataStructure):
  def create_empty(self, environment, policy):
    return policy

@register_supertype(DataStructure)
class RunningNormData(DataStructure):
  def create_empty(self, environment, policy):
    return RunningNormDataStruct()

class RunningNormDataStruct:
  def __init__(self):
    # Welford's Algorithm is a numerically stable algorithm for computing running standard deviations
    self.welfords = mlca.helpers.statistics.welfords_std.Welford()

  def update(self, num):
    self.welfords.update(num)

  def mean(self):
    return self.welfords.mean

  def std(self):
    return self.welfords.std

@register_supertype(DataStructure)
class VariableBuffer(DataStructure):
  def create_empty(self, environment, policy):
    return VariableBufferStruct(
      TspParams.current().MAX_VARIABLE_BUFFER_SIZE
    )

class VariableBufferStruct:
  def __init__(self, MAX_BUFFER_SIZE):
    self.buffer: List[Any] = []
    self.MAX_BUFFER_SIZE = MAX_BUFFER_SIZE

  def update(self, var):
    self.buffer.append(var)
    self.buffer = self.buffer[-self.MAX_BUFFER_SIZE:]

@register_supertype(DataStructure)
class FeatureVectorRunningNormData(DataStructure):
  def create_empty(self, environment, policy):
    return FeatureVectorRunningNormDataStruct()

class FeatureVectorRunningNormDataStruct:
  def __init__(self):
    # Welford's Algorithm is a numerically stable algorithm for computing running standard deviations
    self.item_welfords = mlca.helpers.statistics.welfords_std.Welford()

  def update(self, feature_vector):
    for item in feature_vector.cpu():
      self.item_welfords.update()

  def mean(self):
    means = np.array([w.mean for w in self.item_welfords])
    return torch.tensor(means, device=DefaultDevice.current())

  def std(self):
    stds = np.array([w.std for w in self.item_welfords])
    return torch.tensor(stds, device=DefaultDevice.current())

def get_action_space_size(action_space):
  if action_space.__class__.__name__ == "Box":
    return action_space.shape[0]
  else:
    return action_space.n
