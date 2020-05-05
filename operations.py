"""
The operations that synthesized programs consist of.

Every operation implements a .execute method, along with a variety of 
toggles that control where it's allowed in the program.

Operations can be registered to an operation list, such as the list of 
reward combiner operations or the list of intrinsic reward ("curiosity") 
operations. These lists are configured in operations_list.py
"""

from __future__ import annotations

import mlca.program_types as program_types
from mlca.test_synthesized_programs_experiments import TspParams
from mlca.helpers.config import DefaultDevice
from functools import lru_cache
from typing import List, Mapping, Set, ClassVar, Optional, Tuple
import math

import torch
import torch.nn.functional as F

# ==================
# Base Classes
# ==================
class Operation:
  # Static vars, manually configured for each operation
  # The ProgramTypes that this operation takes
  input_program_types: ClassVar[Tuple[program_types.Type, ...]]
  # Can this operation be optimized so it's output reaches a bounded minimum (ex. zero)?
  can_optimize_to_bounded_minimum: ClassVar[bool]
  # Is this operation commutative?
  commutative: ClassVar[bool] = False
  # The operation's output type
  output_type: ClassVar[program_types.Type] = None
  # Does this operation have a bounded minimum? Ex. SqrtAbs has a min of 0
  might_have_bounded_minimum: ClassVar[bool] = False
  # Does this operation create gradients? Ex. a neural network
  creates_gradients: ClassVar[bool] = False
  # Does this operation make gradients flow backwards? Ex. a backprop operation
  generates_backward_gradients: ClassVar[bool] = False
  # Can this operation only be used once per program?
  can_only_use_once: ClassVar[bool] = False
  # Does this operation propagate gradients? (Ex. .detach() does not)
  propagates_gradients: ClassVar[bool] = True
  # Does this operation let gradients to only a subset of its inputs?
  propagates_gradient_to_input: ClassVar[Optional[List[bool]]] = None
  # Does this operation selectively propagate gradients to some inputs but not others?
  propagates_gradients_to_input: ClassVar[Optional[Tuple[bool, ...]]] = None
  # Does this operation mutate a datastructure?
  mutates_datastructure: ClassVar[bool] = False
  # Is this operation only permitted if it combines operations coming from different inputs?
  # Ex. used to avoid repeatedly running a NN on the output of a same piece of data, because that wouldn't add on any more information.
  require_creates_larger_input_set: ClassVar[Optional[bool]] = None
  # Are only a few inputs considered when requiring the larger input set?
  input_mask_for_larger_set: ClassVar[Optional[List[bool]]] = None
  # Does this operation have a .value_to_add_to_loss(x) operation?
  add_to_loss: ClassVar[bool] = False
  # Is this operation forbidden in the update phase?
  forbiden_in_update_phase: ClassVar[bool] = False

  # Static vars, automatically set
  # This operation's unique ID number, used for avoiding duplicates in the program synthesizer
  operation_number: ClassVar[int] = -1

  # Instance vars
  # The operation's inputs
  inputs: List[Operation]
  # Is this operation representing a datastructure variable?
  is_data_structure: bool
  # Is this operation a variable that's only allowed in the update phase?
  only_allowed_in_update: bool
  # What inputs are used to compute this operation or its ancestors?
  input_set: Set[Operation]
  # Can this operation's output be optimized to a bounded min, based on its input?
  cached_can_optimize_to_bounded_minimum: bool
  # How many time is this operation used in the current program?
  num_uses_in_program: int

  def __init__(self, *inputs, cached_output_type = None):
    self.inputs = inputs
    self.check_program_types_match_signature(inputs, self.input_program_types)
    self.is_data_structure = False
    self.only_allowed_in_update = False # xxx: Don't try to manually override this; instead you might want to set mutates_datastructure = True
    self.input_set = self._join_input_sets(inputs)
    
    self.cached_can_optimize_to_bounded_minimum = self.can_optimize_to_bounded_minimum_fn(inputs)

    if cached_output_type is None:
      self.cached_output_type = self.output_type_fn(
              [i.cached_output_type for i in inputs])
    else: 
      self.cached_output_type = cached_output_type

    # Helpers for storing data about current program
    self.num_uses_in_program = 0
  
  @staticmethod
  def check_program_types_match_signature(inputs, input_program_types: List[program_types.Type]):
    assert len(inputs) == len(input_program_types)

    for input_node, input_type in zip(inputs, input_program_types):
      assert program_types.equal_or_supertype(
        input_node.cached_output_type.__class__, 
        input_type), ("wanted", input_type, "got", input_node.output_type)
    
  @staticmethod
  def _join_input_sets(inputs: List[Operation]):
    return set().union(*(i.input_set for i in inputs))

  @classmethod
  def propagates_gradients_to_input_i(cls, i: int) -> int:
    if cls.propagates_gradient_to_input is not None:
      return cls.propagates_gradients_to_input[i]
    else:
      return cls.propagates_gradients

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return cls.can_optimize_to_bounded_minimum 

  @classmethod
  def output_type_fn(cls, input_types: List[program_types.Type]):
    output_type_class = cls._get_output_type_class_from_input_types(input_types)
    if output_type_class == INVALID_INPUTS:
      return INVALID_INPUTS
    else: 
      assert issubclass(output_type_class, program_types.Type)

      output_type = output_type_class()

      output_type.has_gradients = cls.creates_gradients or \
        (cls.propagates_gradients and \
         any(i.has_gradients for i in input_types))

      output_type.is_constant = all(i.is_constant for i in input_types)

      output_type.might_have_bounded_minimum = cls.might_have_bounded_minimum or \
        all(i.might_have_bounded_minimum for i in input_types)

      return output_type

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_type: program_types.Type):
    return cls.output_type

  """Helper that forbids duplicate inputs. Note that subclasses need to explicitly add this if they want it."""
  @classmethod
  def _forbid_inputs_of_same_type_as_this_operation(cls, inputs: List[Operation]):
    return cls._forbid_inputs_of_type(inputs, cls)

  @classmethod
  def _forbid_inputs_of_type(cls, inputs: List[Operation], match_class):
    return not any(i.__class__ == match_class for i in inputs)

  @classmethod
  def _forbid_descendant_of_type(cls, inputs: List[Operation], match_class):
    return not any(
      i.__class__ == match_class or cls._forbid_descendant_of_type(i.inputs, match_class)
      for i in inputs)

  @classmethod
  def _forbid_duplicate_inputs(cls, inputs: List[Operation]):
    return not any(a == b for a, b in zip(inputs[1:], inputs[:-1]))

  @classmethod
  def _forbid_all_inputs_constant(cls, inputs: List[Operation]):
    return not all(i.cached_output_type.is_constant for i in inputs)

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    if not cls._forbid_all_inputs_constant(inputs):
      return False
    elif len(inputs) <= 1:
      return True
    else:
      # Check larger input sets
      assert cls.require_creates_larger_input_set is not None, ("You need to manually specify require_creates_larger_input_set for any Operation with >1 input", cls)

      inputs_for_larger_set = inputs if cls.input_mask_for_larger_set is None else [i for i, mask in zip(inputs, cls.input_mask_for_larger_set) if mask]

      return not \
        (cls.require_creates_larger_input_set and
         len(cls._join_input_sets(inputs_for_larger_set)) <= max(len(i.input_set) for i in inputs_for_larger_set))

  def __repr__(self):
    # Put each child on a separate line
    n = ("\n\t".join(
      # Insert a \t in front of each line of the children
      "\n\t".join(str(s).split("\n")) 
        for s 
        in self.inputs))

    return f"{self.__class__.__name__} (\n\t{n } )"

  @lru_cache(maxsize=None)
  def program_number(self):
    c = " ".join(i.program_number() for i in self.inputs)
    assert len(str(self.operation_number)) <= 3, self.operation_number
    return f"{str(self.operation_number).zfill(3)} {c}"

class Variable(Operation):
  might_have_bounded_minimum = True
  can_optimize_to_bounded_minimum = False

  def __init__(
          self, var_type, variable_number, name=None,
          is_data_structure=False, only_allowed_in_update=False, is_optimizer=False,
          can_only_use_once=False, is_constant=False, must_be_used=False, short_name=None):
    self.var_type = var_type
    self.output_type = var_type
    self.cached_output_type = var_type
    self.cached_can_optimize_to_bounded_minimum = False
    self.name = name
    self.short_name = short_name
    self.is_data_structure = is_data_structure
    self.is_optimizer = is_optimizer
    self.only_allowed_in_update = only_allowed_in_update
    self.can_only_use_once = can_only_use_once
    self.must_be_used = must_be_used
    self.output_type.is_constant = is_constant

    self.input_set = set([self])

    self.operation_number = f"V{variable_number}"
    self.inputs = []
    self.num_uses_in_program = 0

    assert not var_type.must_be_constant or var_type.is_constant

  def __repr__(self):
    n = self.name if self.name else ""
    return n

# ==================
# Policy
# ==================

class Policy(Operation):
  input_program_types = (program_types.ImageTensor, program_types.Policy)
  output_prorogram_type = program_types.FeatureVectorActionSpace
  require_creates_larger_input_set = False

  def execute(self, input_values, profiler=None, i_episode=None):
    state, policy = input_values
    assert not TspParams.current().AGENT_RECURRENT, "The 'Policy' operation does not support recurrent policies." 
    value, action, action_log_prob, recurrent_hidden_states = policy.act(
      state, None, None)
    return action.detach()
  
  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == program_types.NonNegativeNumber and input_types[1].__class__ == program_types.NonNegativeNumber:
      return program_types.NonNegativeNumber
    else:
      return program_types.RealNumber

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and super().inputs_allowed(inputs)

# ==================
# Arithmetic
# ==================

class Add(Operation):
  input_program_types = (program_types.RealNumber, program_types.RealNumber)
  commutative = True
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0] + .2 *  input_values[1]
  
  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == program_types.NonNegativeNumber and input_types[1].__class__ == program_types.NonNegativeNumber:
      return program_types.NonNegativeNumber
    else:
      return program_types.RealNumber

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and super().inputs_allowed(inputs)

class Sin(Operation):
  input_program_types = (
    program_types.RealNumber, 
  )
  output_type = program_types.RealNumber
  
  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.sin(input_values[0])

class Max(Operation):
  input_program_types = (
    program_types.RealNumber, program_types.RealNumber)
  require_creates_larger_input_set = True
  commutative = True

  def execute(self, input_values, profiler=None, i_episode=None):
    a, b = input_values
    return max(a, b)

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == program_types.NonNegativeNumber and input_types[1].__class__ == program_types.NonNegativeNumber:
      return program_types.NonNegativeNumber
    else:
      return program_types.RealNumber

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and super().inputs_allowed(inputs)


class Min(Operation):
  input_program_types = (
      program_types.RealNumber, program_types.RealNumber)
  require_creates_larger_input_set = True
  commutative = True

  def execute(self, input_values, profiler=None, i_episode=None):
    a, b = input_values
    return min(a, b)

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == program_types.NonNegativeNumber and input_types[1].__class__ == program_types.NonNegativeNumber:
      return program_types.NonNegativeNumber
    else:
      return program_types.RealNumber

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and super().inputs_allowed(inputs)


class IntrinsicExtrinsicWeightedNormalizedSum(Operation):
  input_program_types = (
    program_types.RealNumber, # a's_weight
    program_types.RealNumber, # a
    program_types.RealNumber, # b's_weight
    program_types.RealNumber,)# b
  input_program_names = (
    "a", "w_a", "b", "w_b"
  )
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    w_a, a, w_b, b = input_values
    w_a, w_b = w_a.abs(), w_b.abs()
    w_norm_a = w_a / (w_a + w_b)
    w_norm_b = w_b / (w_a + w_b)
    return w_norm_a * a + w_norm_b * b

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == program_types.NonNegativeNumber \
      and input_types[1].__class__ == program_types.NonNegativeNumber \
        and  input_types[2].__class__ == program_types.NonNegativeNumber \
          and input_types[3].__class__ == program_types.NonNegativeNumber:
      return program_types.NonNegativeNumber
    else:
      return program_types.RealNumber

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    w_a, a, w_b, b = inputs
    return cls._forbid_duplicate_inputs(inputs) and \
      type(a) is Variable and a.name == "intrinsic_reward" and \
      type(b) is Variable and b.name == "extrinsic_reward" and \
      super().inputs_allowed(inputs)

WeightedNormalizedSSum = IntrinsicExtrinsicWeightedNormalizedSum

class RunningNorm(Operation):
  input_program_types = (program_types.RealNumber,
   program_types.RunningNormData)
  output_type = program_types.RealNumber
  propagates_gradients = False
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    num, data = input_values

    for n in num.detach():
      data.update(n)

    std = data.std()

    if std == 0:
      return torch.zeros(num.shape, device=DefaultDevice.current())
    else: 
      return (num - data.mean()) /  std

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and super().inputs_allowed(inputs)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class RunningNormDontCenter(Operation):
  input_program_types = (program_types.RealNumber,
                         program_types.RunningNormData)
  output_type = program_types.RealNumber
  propagates_gradients = False
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    num, data = input_values

    for n in num.detach():
      data.update(n)

    std = data.std()

    if std == 0:
      return torch.zeros(num.shape, device=DefaultDevice.current())
    else:
      return num / std # Note we DO NOT divide by the mean here (difference with RunningNorm)

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and super().inputs_allowed(inputs)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class VariableAsBufferCombined(Operation):
  """
  Takes a variable and makes  'buffer' list from it that stores the last
  `TspParams.current().MAX_VARIABLE_BUFFER_SIZE` values.
  Works properly with batches by splitting apart each timestep
  """
  input_program_types = (program_types.Type, program_types.VariableBuffer)
  can_optimize_to_bounded_minimum = False  
  require_creates_larger_input_set = True
  propagates_gradients = False

  def execute(self, input_values, profiler=None, i_episode=None):
    v, buffer = input_values
    v = v.detach()
    for i in range(len(v)):
      buffer.update(v[i: i+1])
    return buffer.buffer

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    t = type(input_types[0])

    if program_types.equal_or_supertype(t, program_types.FeatureVector32):
      return program_types.ListFeatureVector32
    elif program_types.equal_or_supertype(t, program_types.FeatureVectorActionSpace):
      return program_types.ListFeatureVectorActionSpace
    elif program_types.equal_or_supertype(t, program_types.ImageTensor):
      return program_types.ListImageTensor
    elif program_types.equal_or_supertype(t, program_types.RealNumber):
      return program_types.ListRealNumber
    else:
      return INVALID_INPUTS

# Deprecated, use VariableAsBufferSeparate
class VariableAsBuffer(Operation):
  """
  Takes a variable and makes  'buffer' list from it that stores the last
  `TspParams.current().MAX_VARIABLE_BUFFER_SIZE` values.
  """
  input_program_types = (program_types.Type, program_types.VariableBuffer)
  can_optimize_to_bounded_minimum = False  
  require_creates_larger_input_set = True
  propagates_gradients = False

  def execute(self, input_values, profiler=None, i_episode=None):
    var, buffer = input_values
    buffer.update(var.detach())
    return buffer.buffer

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    t = type(input_types[0])

    if program_types.equal_or_supertype(t, program_types.FeatureVector32):
      return program_types.ListFeatureVector32
    elif program_types.equal_or_supertype(t, program_types.FeatureVectorActionSpace):
      return program_types.ListFeatureVectorActionSpace
    elif program_types.equal_or_supertype(t, program_types.ImageTensor):
      return program_types.ListImageTensor
    elif program_types.equal_or_supertype(t, program_types.RealNumber):
      return program_types.ListRealNumber
    else:
      return INVALID_INPUTS

# Note: Intentionally unregistered
class FeatureVectorRunningNorm(Operation):
  # DISABLED for now because we're unsure if worth the amount of programs it creates
  short_name = "RunningNorm"
  input_program_types = (program_types.FeatureVector, 
   program_types.FeatureVectorRunningNormData
  )
  propagates_gradients = False
  require_creates_larger_input_set = True
  # can_optimize_to_bounded_minimum = False

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and super().inputs_allowed(inputs)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    return input_types[0].__class__

  def execute(self, input_values, profiler=None, i_episode=None):
    feature_vector, data = input_values

    data.update(num)

    std = data.std()
    if (std == 0).any():
      return torch.zeros(feature_vector.shape, device=DefaultDevice.current())
    else: 
      return (feature_vector - data.mean()) /  std

# "Fixed" means that this actually performs regression, rather than just finding the
# nearest neighbor as "NearestNeighborRegressor" did
class NearestNeighborRegressorFixed(Operation):
  short_name = "NearestNeighborRegressor"
  require_creates_larger_input_set = True
  input_program_types = (program_types.FeatureVector32, # target
                         program_types.FeatureVector32, # query
                         program_types.NearestNeighborRegressor)
  input_program_names = ("target", "query", None)
  output_type = program_types.FeatureVector32
  can_optimize_to_bounded_minimum = False
  require_creates_larger_input_set = True
  mutates_datastructure = True
  forbiden_in_update_phase = True
  propagates_gradients = False

  def execute(self, input_values, profiler=None, i_episode=None):
    targets, queries, knn = input_values
    targets, queries = targets.detach(), queries.detach()

    predictions = [knn.predict(query) for query in queries]
    for query, target in zip(queries, targets):
      knn.add(query, target)

    return torch.stack(predictions)

class NearestNeighborSmall(Operation):
  short_name = "NearestNeighbor"
  require_creates_larger_input_set = True
  input_program_types = (program_types.FeatureVector32, # stream
                         program_types.FeatureVector32, # query
                         program_types.NearestNeighborSmall)
  input_program_names = ("stream", "query", None)
  output_type = program_types.FeatureVector32
  can_optimize_to_bounded_minimum = False
  require_creates_larger_input_set = True
  mutates_datastructure = True
  forbiden_in_update_phase = True
  propagates_gradients = False

  def execute(self, input_values, profiler=None, i_episode=None):
    targets, queries, knn = input_values
    targets, queries = targets.detach(), queries.detach()

    predictions = [knn.predict(query) for query in queries]
    for target in targets:
      knn.add(target)
    # predictions = []
    # for target, query in zip(targets, queries):
    #   predictions.append(knn.predict(query))
    #   knn.add(target)

    return torch.stack(predictions).to(DefaultDevice.current())

# Don't use this anymore, this is not actually a regressor
class NearestNeighborLarge(Operation):
  short_name = "NearestNeighbor"
  require_creates_larger_input_set = True
  input_program_types = (program_types.FeatureVector32, # stream
                         program_types.FeatureVector32, # query
                         program_types.NearestNeighborLarge)
  input_program_names = ("stream", "query", None)
  output_type = program_types.FeatureVector32
  can_optimize_to_bounded_minimum = False
  require_creates_larger_input_set = True
  mutates_datastructure = True
  forbiden_in_update_phase = True
  propagates_gradients = False

  def execute(self, input_values, profiler=None, i_episode=None):
    targets, queries, knn = input_values
    targets, queries = targets.detach(), queries.detach()

    predictions = [knn.predict(query) for query in queries]
    for target in targets:
      knn.add(target)
    # predictions = []
    # for target, query in zip(targets, queries):
    #   predictions.append(knn.predict(query))
    #   knn.add(target)

    return torch.stack(predictions).to(DefaultDevice.current())

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) \
      and inputs[0] != inputs[1] \
      and super().inputs_allowed(inputs)

# For backwards compatibility - 
NearestNeighborRegressor = NearestNeighborLarge
NearestNeighbor = NearestNeighborRegressor

# # Intentionally not registering, because can't correctly backprop gradients. 
class LSTM(Operation):
  input_program_types = (program_types.FeatureVector32,
                         program_types.LSTM32)
  output_type = program_types.FeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    # Note that the lstm internally stores the h feature vector
    x, lstm = input_values
    return lstm(x)

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) \
      and super().inputs_allowed(inputs)


class SubtractOneTenth(Operation):
  input_program_types = (program_types.RealNumber, )
  output_type = program_types.RealNumber
  require_creates_larger_input_set = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0] - .1

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and super().inputs_allowed(inputs)

def get_batch_size_for_current_experiment():
  if TspParams.current().REAL_BATCH_REWARD_COMPUTATION:
    if TspParams.current().SHARE_CURIOSITY_MODULE_IN_TRIAL:
      if TspParams.current().STEPS_PER_CURIOSITY_UPDATE is None:
        return TspParams.current().PPO_FRAMES_PER_PROC * TspParams.current().NUM_ROLLOUTS_PER_TRIAL
      else: 
        return TspParams.current().STEPS_PER_CURIOSITY_UPDATE 
    else:
      if TspParams.current().STEPS_PER_CURIOSITY_UPDATE is None:
        return TspParams.current().PPO_FRAMES_PER_PROC
      else: 
        return TspParams.current().STEPS_PER_CURIOSITY_UPDATE 
  else:
    return 1

class NormalDistribution(Operation):
  input_program_types = tuple()  # type: ignore
  output_type = program_types.RealNumber
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.normal(
      torch.zeros(get_batch_size_for_current_experiment()), 
      torch.ones(get_batch_size_for_current_experiment())
    ).to(DefaultDevice.current())
  
# Intentionally not @register_operation
class ConstantZero(Operation):
  input_program_types = tuple() # type: ignore
  output_type = program_types.RealNumber
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.zeros(get_batch_size_for_current_experiment(), device=DefaultDevice.current())

# Intentionally not @register_operation
class Identity(Operation):
  input_program_types = (program_types.Type, )
  output_type = program_types.RealNumber
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0]

# Intentionally not @register_operation
class ConstantOne(Operation):
  input_program_types = tuple() # type: ignore
  output_type = program_types.RealNumber
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.ones(get_batch_size_for_current_experiment(), device=DefaultDevice.current())

# Intentionally not @register_operation
class ConstantNegativeOne(Operation):
  input_program_types = tuple()  # type: ignore
  output_type = program_types.RealNumber
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return -torch.ones(get_batch_size_for_current_experiment(), device=DefaultDevice.current())

class Subtract(Operation):
  input_program_types = (program_types.RealNumber, program_types.RealNumber)
  output_type = program_types.RealNumber
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0] - input_values[1]

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and super().inputs_allowed(inputs)

class Multiply(Operation):
  input_program_types = (program_types.RealNumber, program_types.RealNumber)
  output_type = program_types.RealNumber
  commutative = True
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0] * input_values[1]

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum or inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and super().inputs_allowed(inputs)

class Square(Operation):
  input_program_types = (program_types.RealNumber, )
  output_type = program_types.RealNumber

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0] * input_values[0]

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class Clip(Operation):
  input_program_types = (program_types.RealNumber, program_types.RealNumber)
  input_program_names = ("value", "bound")
  output_type = program_types.RealNumber
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.min(torch.max(input_values[0], -input_values[1]), input_values[1])

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum or inputs[1].cached_output_type.is_constant


class SquareRootAbs(Operation):
  short_name = "Sqrt(Abs(x))"
  input_program_types = (program_types.RealNumber, )
  output_type = program_types.NonNegativeNumber
  might_have_bounded_minimum = True

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) \
      and super()._forbid_inputs_of_type(inputs, L2Distance) \
      and super()._forbid_inputs_of_type(inputs, L2Norm) \
      and super().inputs_allowed(inputs)

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0].float().abs().sqrt()
    # return torch.sqrt(torch.abs(input_values[0]))

# ==================
# Neural Networks
# ==================

class FullyConnectedNetworkTwo32to32(Operation):
  short_name = "NN: 𝔽 x 𝔽 → 𝔽"
  input_program_types = (program_types.FeatureVector32, program_types.FeatureVector32,
                         program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32)
  input_mask_for_larger_set = [True, True, False]
  output_type = program_types.FeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    combined_input = torch.cat((input_values[0], input_values[1]), 1)
    return input_values[2](combined_input)

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    a, b, nn = inputs
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and \
      a != b and \
      a.program_number() > b.program_number() and \
      super().inputs_allowed(inputs)

class FullyConnectedNetworkTwo32toActionSpace(Operation):
  short_name = "NN: 𝔽 x 𝔽 → 𝔸"
  input_program_types = (program_types.FeatureVector32, program_types.FeatureVector32,
                         program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace)
  input_mask_for_larger_set = [True, True, False]
  output_type = program_types.FeatureVectorActionSpace
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    combined_input = torch.cat((input_values[0], input_values[1]), 1)
    return input_values[2](combined_input)

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    a, b, nn = inputs
    return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and \
      a != b and \
      a.program_number() > b.program_number() and \
      super().inputs_allowed(inputs)

class FullyConnectedNetwork32toActionSpace(Operation):
  short_name = "NN: 𝔽 → 𝔸"
  input_program_types = (program_types.FeatureVector32,
                         program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace)
  output_type = program_types.FeatureVectorActionSpace
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[1](input_values[0])

class FullyConnectedNetworkActionSpaceto32(Operation):
  short_name = "NN: 𝔸 → 𝔽"
  input_program_types = (program_types.FeatureVectorActionSpace,
                         program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32) 
  output_type = program_types.FeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[1](input_values[0])

class PredictFeatureVector32FromFeatureVector32(Operation):
  short_name = "Predict Target From Query"
  input_program_types = (
    program_types.FeatureVector32,
    program_types.FeatureVector32,
    program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVector32)
  input_program_names = (
    "query", "target", None
  )
  propagates_gradients_to_input = (
    True, False, True
  )
  output_type = program_types.FeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True
  add_to_loss = True
  generates_backward_gradients = True
  mutates_datastructure = True 
  forbiden_in_update_phase = True

  def execute(self, input_values, profiler=None, i_episode=None):
    query, target, nn = input_values
    prediction = nn(query)
    return prediction

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and \
        super().inputs_allowed(inputs)

  def value_to_add_to_loss(self, input_values, execute_output_value, profiler=None, i_episode=None):
    query, target, nn = input_values
    prediction = execute_output_value
    err = torch.norm(prediction - target.float().detach(),
                     p=2, dim=1) / math.sqrt(target.shape[1])
    return err

class ConditionalVAEReconstruction(Operation):
  short_name = "NN: 𝕊 → 𝔽"
  input_program_types = (
    program_types.FeatureVector32, # VAE x
    program_types.FeatureVector32, # VAE c
    program_types.NeuralNetworkWeightsConditionalVAE)
  input_program_names = ("input", "conditional", None)  
  output_type = program_types.FeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    nn = input_values[1]
    recon_x, means, log_var, z = nn(input_values[0])
    return recon_x

# class ConditionalVAEEncoding(Operation):
#   short_name = "NN: 𝕊 → 𝔽"
#   input_program_types = (
#     program_types.FeatureVector32, # VAE x
#     program_types.FeatureVector32, # VAE c
#     program_types.NeuralNetworkWeightsConditionalVAE)
#   input_program_names = ("input", "conditional", None)  
#   output_type = program_types.FeatureVector32
#   creates_gradients = True
#   require_creates_larger_input_set = True
#   can_optimize_to_bounded_minimum = True

#   def execute(self, input_values, profiler=None, i_episode=None):
#     nn = input_values[1]
#     recon_x, means, log_var, z = nn(input_values[0])
#     return recon_x

class CNN(Operation):
  short_name = "NN: 𝕊 → 𝔽"
  input_program_types = (program_types.ImageTensor,
                         program_types.NeuralNetworkWeightsObservationToFeatureVector32)
  output_type = program_types.FeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    nn = input_values[1]
    o = nn(input_values[0])
    return o

class CNNDetachOutput(Operation):
  short_name = "NN: 𝕊 → 𝔽 + Detach"
  input_program_types = (program_types.ImageTensor,
                         program_types.NeuralNetworkWeightsObservationToFeatureVector32)
  output_type = program_types.FeatureVector32
  creates_gradients = False
  propagates_gradients = False
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    nn = input_values[1]
    o = nn(input_values[0]).detach()
    return o

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    # Also have CNNWithoutGradients, so forbid DetachFeatureVector(CNN(x))
    return super()._forbid_descendant_of_type(inputs, DetachFeatureVector) \
      and super()._forbid_descendant_of_type(inputs, CNNWithoutGradients) \
      and super().inputs_allowed(inputs)

CNNWithoutGradients = CNNDetachOutput

class CNNEnsemble(Operation):
  short_name = "NN: 𝕊 → [𝔽]"
  input_program_types = (program_types.ImageTensor,
                         program_types.EnsembleWeightsImageTo32)
  output_type = program_types.ListFeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[1](input_values[0])

class FullyConnectedNetworkEnsemble32To32(Operation):
  short_name = "NN Ensemble 32 → 32"
  input_program_types = (program_types.FeatureVector32,
                         program_types.EnsembleWeights32To32)
  output_type = program_types.ListFeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[1](input_values[0])

class FullyConnectedNetworkEnsembleTwo32To32(Operation):
  short_name = "NN Ensemble 64 → 32"
  input_program_types = (program_types.FeatureVector32, program_types.FeatureVector32,
                         program_types.EnsembleWeightsTwo32To32)
  input_mask_for_larger_set = [True, True, False]
  output_type = program_types.ListFeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    combined_input = torch.cat((input_values[0], input_values[1]), 1)
    return input_values[2](combined_input)
  
  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    a, b, nn = inputs
    return a != b and \
      a.program_number() > b.program_number() and \
      super().inputs_allowed(inputs)


class FullyConnectedNetworkEnsemble32AndActionTo32(Operation):
  short_name = "NN: 𝔽 x 𝔸 → [𝔽]" # NN Ensemble 32 + #Actions → 32"
  input_program_types = (program_types.FeatureVector32, program_types.FeatureVectorActionSpace,
                         program_types.EnsembleWeights32AndActionTo32)
  input_mask_for_larger_set = [True, True, False]
  output_type = program_types.ListFeatureVector32
  creates_gradients = True
  require_creates_larger_input_set = True
  can_optimize_to_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    combined_input = torch.cat((input_values[0], input_values[1]), 1)
    return input_values[2](combined_input)

class AddToLoss(Operation):
  short_name = "Add To Loss"
  input_program_types = (program_types.RealNumber, )
  output_type = program_types.Void
  can_optimize_to_bounded_minimum = False
  add_to_loss = True
  generates_backward_gradients = True

  mutates_datastructure = True
  forbiden_in_update_phase = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return None

  def value_to_add_to_loss(self, input_values, execute_output_value, profiler=None, i_episode=None):
    return input_values[0]

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0].cached_output_type.has_gradients \
      and inputs[0].might_have_bounded_minimum \
      and not inputs[0].cached_can_optimize_to_bounded_minimum \
      and super().inputs_allowed(inputs)

# Intentionally not registering 
class MinimizeValue(Operation):
  input_program_types = (program_types.RealNumber, program_types.Optimizer)
  output_type = program_types.Void
  can_optimize_to_bounded_minimum = False

  mutates_datastructure = True
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    loss, optimizer = input_values

    loss = loss.mean()

    optimizer.zero_grad()
    if profiler is not None:
        profiler.tick(i_episode, "MinimizeValue: zero grad")

    loss.backward()
    if profiler is not None:
        profiler.tick(i_episode, "MinimizeValue: backward")

    optimizer.step()
    if profiler is not None:
        profiler.tick(i_episode, "MinimizeValue: step")

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0].cached_output_type.has_gradients \
      and inputs[0].might_have_bounded_minimum \
      and not inputs[0].cached_can_optimize_to_bounded_minimum \
      and super().inputs_allowed(inputs)

# # class MaximizeValue(Operation):
#   input_program_types = (program_types.RealNumber, program_types.Optimizer)
#   output_type = program_types.Void

#   mutates_datastructure = True
#   require_creates_larger_input_set = True

#   def execute(self, input_values, profiler=None, i_episode=None):
#     loss, optimizer = input_values

#     optimizer.zero_grad()
#     neg_loss = - loss
#     neg_loss.backward()
#     # for param in self.q_net.parameters():
#     #     if param.grad is not None:
#     #         param.grad.data.clamp_(-1, 1)
#     optimizer.step()

#   @classmethod
#   def inputs_allowed(cls, inputs: List[Operation]):
#     return inputs[0].cached_output_type.has_gradients and super().inputs_allowed(inputs)

# ==================
# Feature Vector
# ==================
# # class Softmax(Operation):
#   Added SoftmaxNLL instead
# 
#   input_program_types = (program_types.FeatureVector, )
#   might_have_bounded_minimum = True

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return F.softmax(input_values[0])

#   @classmethod
#   def inputs_allowed(cls, inputs: List[Operation]):
#     return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and super()._forbid_inputs_of_type(inputs, Sigmoid) and super().inputs_allowed(inputs)

#   @classmethod
#   def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
#     return input_types[0].__class__

# # class Sigmoid(Operation):
#   Seems unecessary
#   input_program_types = (program_types.FeatureVector, )

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return F.sigmoid(input_values[0])

#   @classmethod
#   def inputs_allowed(cls, inputs: List[Operation]):
#     return super()._forbid_inputs_of_same_type_as_this_operation(inputs) and super()._forbid_inputs_of_type(inputs, Softmax) and super().inputs_allowed(inputs)

#   @classmethod
#   def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
#     return input_types[0].__class__

# class SumFeatureVector(Operation):
#   input_program_types = (program_types.FeatureVector, )
#   output_type = program_types.RealNumber

#   @classmethod
#   def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
#     return inputs[0].cached_can_optimize_to_bounded_minimum

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return torch.sum(input_values[0])

# # class Variance(Operation):
#   input_program_types = (program_types.FeatureVector, )
#   output_type = program_types.NonNegativeNumber
#   might_have_bounded_minimum = True

#   @classmethod
#   def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
#     return inputs[0].cached_can_optimize_to_bounded_minimum

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return torch.var(input_values[0])

class L2Norm(Operation):
  input_program_types = (program_types.FeatureVector, )
  output_type = program_types.NonNegativeNumber
  might_have_bounded_minimum = True

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

  def execute(self, input_values, profiler=None, i_episode=None):
    x = input_values[0]
    return torch.norm(x.float(), p=2, dim=1) / math.sqrt(x.shape[1])

class L2Distance(Operation):
  short_name = "L2 Distance"
  input_program_types = (program_types.FeatureVector,
                         program_types.FeatureVector)
  might_have_bounded_minimum = True
  require_creates_larger_input_set = True
  commutative = True

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum
  
  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == input_types[1].__class__:
      return program_types.NonNegativeNumber
    else:
      return INVALID_INPUTS

  def execute(self, input_values, profiler=None, i_episode=None):
    x, y = input_values
    return torch.norm(x - y, p=2, dim=1) / math.sqrt(x.shape[1])

# Don't register, because this has the wrong type signature. Fixed is below.
class SoftmaxAndNLL(Operation):
  short_name = "Action Prediction Loss" 
  input_program_types = (program_types.FeatureVector,
                         program_types.FeatureVectorActionSpace)
  input_program_names = ("prediction", "target")
  output_type = program_types.NonNegativeNumber
  require_creates_larger_input_set = True
  might_have_bounded_minimum = True

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  def execute(self, input_values, profiler=None, i_episode=None):
    prediction, target = input_values
    assert prediction.requires_grad 

    if TspParams.current().CONTINUOUS_ACTION_SPACE: 
      if TspParams.current().FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS:
        padded_target = torch.cat(
          (target, 
          torch.zeros(
            target.shape[0], prediction.shape[1] - target.shape[1],
            device=DefaultDevice.current())),
          dim=1
        )
        return torch.norm(
          prediction - padded_target
        , p=2, dim=1) / math.sqrt(prediction.shape[1])
      else:
        return torch.norm(prediction[:, :target.shape[1]] - target, p=2, dim=1) / math.sqrt(prediction.shape[1])
    else:
      # NOTE: target is assumed to be a 1-hot encoding. Thus,
      # we find the index it represents by taking the argmax.
      if input_values[1].shape == (prediction.shape[0],):
        # XXX: DEPRECATED Path
        target_num = target.long()
      else: 
        target_num = target.argmax(dim=1).long()

      return F.cross_entropy(prediction, target_num, reduction="none")

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and \
      inputs[0].cached_output_type.has_gradients and \
      super().inputs_allowed(inputs)


# This is the "fixed" version of SoftmaxAndNLL that requires both inputs to 
# have the same dimension
class SoftmaxAndNLLFixed(Operation):
  short_name = "Action Prediction Loss" 
  input_program_types = (program_types.FeatureVectorActionSpace,
                         program_types.FeatureVectorActionSpace)
  input_program_names = ("prediction", "target")
  output_type = program_types.NonNegativeNumber
  require_creates_larger_input_set = True
  might_have_bounded_minimum = True

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  def execute(self, input_values, profiler=None, i_episode=None):
    prediction, target = input_values
    assert prediction.requires_grad

    if TspParams.current().CONTINUOUS_ACTION_SPACE:
      return torch.norm(prediction - target, p=2, dim=1) / math.sqrt(prediction.shape[1])
    else:
      # NOTE: target is assumed to be a 1-hot encoding. Thus,
      # we find the index it represents by taking the argmax.
      if input_values[1].shape == (prediction.shape[0],):
        # XXX: DEPRECATED Path
        target_num = target.long()
      else:
        target_num = target.argmax(dim=1).long()

      return F.cross_entropy(prediction, target_num, reduction="none")

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return inputs[0] != inputs[1] and \
        inputs[0].cached_output_type.has_gradients and \
        super().inputs_allowed(inputs)


class DotProduct(Operation):
  input_program_types = (program_types.FeatureVector,
                         program_types.FeatureVector)
  commutative = True
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.einsum('bs,bs->b', 
      input_values[0],
      input_values[1])

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum or inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == input_types[1].__class__:
      return program_types.RealNumber
    else:
      return INVALID_INPUTS

class AddFeatureVector(Operation):
  short_name = "Add"
  input_program_types = (program_types.FeatureVector, program_types.FeatureVector)
  commutative = True
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0] + input_values[1]

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].__class__ == input_types[1].__class__:
      return input_types[0].__class__
    else:
      return INVALID_INPUTS

# # class SubtractFeatureVector(Operation):
#   input_program_types = (program_types.FeatureVector,
#                          program_types.FeatureVector)
#   require_creates_larger_input_set = True

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return input_values[0] - input_values[1]

#   @classmethod
#   def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
#     return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

#   @classmethod
#   def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
#     if input_types[0].__class__ == input_types[1].__class__:
#       return input_types[0].__class__
#     else:
#       return INVALID_INPUTS

# # class MultiplyFeatureVector(Operation):
#   input_program_types = (program_types.FeatureVector,
#                          program_types.FeatureVector)
#   commutative = True
#   require_creates_larger_input_set = True

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return input_values[0] * input_values[1]

#   @classmethod
#   def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
#     return inputs[0].cached_can_optimize_to_bounded_minimum or inputs[1].cached_can_optimize_to_bounded_minimum

#   @classmethod
#   def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
#     if input_types[0].__class__ == input_types[1].__class__:
#       return input_types[0].__class__
#     else:
#       return INVALID_INPUTS

class DetachFeatureVector(Operation):
  short_name = "Detach"
  input_program_types = (program_types.FeatureVector, )
  output_type = program_types.FeatureVector
  can_optimize_to_bounded_minimum = False
  propagates_gradients = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0].detach()

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    # Also have CNNWithoutGradients, so forbid DetachFeatureVector(CNN(x))
    return inputs[0].cached_output_type.has_gradients \
      and super()._forbid_inputs_of_type(inputs, CNN) \
      and super()._forbid_descendant_of_type(inputs, DetachFeatureVector) \
      and super()._forbid_descendant_of_type(inputs, CNNWithoutGradients) \
      and super().inputs_allowed(inputs)

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    return input_types[0].__class__

# ==================
# List of Real Numbers
# ==================
class RealNumberListMean(Operation):
  short_name = "Mean"
  input_program_types = (program_types.ListRealNumber, )
  output_type = program_types.RealNumber

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.mean(torch.stack(input_values[0]), dim=0)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class RealNumberListLinearRegressionSlope(Operation):
  short_name = "Linear Regression Slope"
  input_program_types = (program_types.ListRealNumber, )
  output_type = program_types.RealNumber

  def execute(self, input_values, profiler=None, i_episode=None):
    input_lists = input_values[0]
    slopes = []
    for i in range(len(input_lists)):
      y_values = input_lists[i]
      if len(y_values) <= 1:
        slope = torch.zeros(1, device=DefaultDevice.current())
      else:
        x_values = torch.arange(len(y_values))
        slope = x_values.float().pinverse() @ y_values
      slopes.append(slope)
    return slopes
  
  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

# Intentionally don't register
class RealNumberListSum(Operation):
  DEPRECATED = True
  short_name = "Sum"
  input_program_types = (program_types.ListRealNumber, )
  output_type = program_types.RealNumber

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.sum(torch.stack(input_values[0]), dim=0)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

# ==================
# List of Feature Vectors
# ==================
class ListVariance(Operation):
  short_name = "Variance"
  input_program_types = (program_types.ListFeatureVector, )
  output_type = program_types.NonNegativeNumber
  might_have_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    # x = [i[0] for i in input_values[0]]
    x = input_values[0]
    mean = torch.mean(torch.stack(x), dim=0)
    assert mean.shape == x[0].shape
    variances = torch.stack([
      torch.norm(i - mean, p=2, dim=1) / math.sqrt(i.shape[0]) 
      for i in x
    ])
    # assert variances.shape == (len(x), input_values[0][0]), (variances.shape, len(x), input_values[0].shape)
    return torch.mean(variances, dim=0)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class MeanOfFeatureVectorList(Operation):
  short_name = "Mean"
  input_program_types = (program_types.ListFeatureVector, )

  @classmethod
  def inputs_allowed(cls, inputs: List[Operation]):
    return super()._forbid_inputs_of_type(inputs, CNNEnsemble) \
      and super().inputs_allowed(inputs)

  def execute(self, input_values, profiler=None, i_episode=None):
    mean = torch.mean(torch.stack(input_values[0]), dim=0)
    assert mean.shape == input_values[0][0].shape
    return mean

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_type):
    return input_type[0].list_contents_type

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class FeatureVectorListL2Norm(Operation):
  short_name = "L2 Norm"
  input_program_types = (program_types.ListFeatureVector, )
  output_type = program_types.ListRealNumber
  might_have_bounded_minimum = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return [torch.norm(x, p=2, dim=1) / x.shape[1] for x in input_values[0]]

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum

class FeatureVectorListAverageL2DistanceToFeatureVector(Operation):
  short_name = "Average Distance"
  input_program_types = (program_types.ListFeatureVector, program_types.FeatureVector)
  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    lst, fv = input_values
    return torch.stack(
      [torch.norm(x - fv, p=2, dim=1) / x.shape[1] for x in lst]
    ).mean(dim=0)

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].list_contents_type == input_types[1].__class__:
      return program_types.NonNegativeNumber
    else:
      return INVALID_INPUTS

class FeatureVectorListMinusFeatureVector(Operation):
  short_name = "Minus"
  input_program_types = (program_types.ListFeatureVector, program_types.FeatureVector)

  require_creates_larger_input_set = True

  def execute(self, input_values, profiler=None, i_episode=None):
    lst, fv = input_values
    return [x - fv for x in lst]

  @classmethod
  def can_optimize_to_bounded_minimum_fn(cls, inputs: List[Operation]):
    return inputs[0].cached_can_optimize_to_bounded_minimum and inputs[1].cached_can_optimize_to_bounded_minimum

  @classmethod
  def _get_output_type_class_from_input_types(cls, input_types: List[program_types.Type]):
    if input_types[0].list_contents_type == input_types[1].__class__:
      return input_types[0].__class__
    else:
      return INVALID_INPUTS

# ==================
# Boolean
# ==================
# class BooleanToBinaryNumber(Operation):
#   input_program_types = (program_types.Boolean,)
#   output_type = program_types.BinaryNumber

#   def execute(self, input_values, profiler=None, i_episode=None):
#     return 1 if input_values[0] else 0

# ==================
# Action
# ==================
class ActionToInteger(Operation):
  # XXX: DEPRECATED
  input_program_types = (program_types.Action,)
  output_type = program_types.Integer
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return torch.tensor(input_values[0], device=DefaultDevice.current()).argmax(dim=1).float()

class ActionToOneHotFeatureVector(Operation):
  # XXX: DEPRECATED
  input_program_types = (program_types.Action,)
  output_type = program_types.FeatureVectorActionSpace
  can_optimize_to_bounded_minimum = False

  @staticmethod
  def one_hot(labels, num_classes, device):
    """Embedding labels to one-hot form.
    Source: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes, device=device)
    return y[labels]

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0].to(DefaultDevice.current()).float()

class ActionTo1DFeatureVector(Operation):
  # XXX: DEPRECATED
  input_program_types = (program_types.Action,)
  output_type = program_types.FeatureVector1
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    # return torch.tensor([input_values], device=DefaultDevice.current())
    return torch.argmax(input_values[0], dim=1)

# ==================
# Counters
# ==================
class IncrementCounter(Operation):
  input_program_types = (program_types.Counter,)
  output_type = program_types.Void

  mutates_datastructure = True

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0].increment()

class GetCounterValue(Operation):
  input_program_types = (program_types.Counter,)
  output_type = program_types.Integer
  can_optimize_to_bounded_minimum = False

  def execute(self, input_values, profiler=None, i_episode=None):
    return input_values[0].get_value()

# The operation number for the next operation that gets registered
cur_operation_number = 0
# Register all operation numbers
for operation in Operation.__subclasses__():
  operation.operation_number = cur_operation_number
  cur_operation_number += 1

INVALID_INPUTS = "INVALID_INPUTS"

def short_name(o):
  return o.short_name if hasattr(o, 'short_name') else o.__name__ 

def latexify(o):
  return str(o).replace("→", "$\\rightarrow$")

if __name__ == "__main__":
  category_name = "intrinsic_reward_programs_v8"
  print("--------------")
  print(category_name)
  print("--------------")

  from mlca.operations_list import OperationsSetList
  for i, operation in enumerate(OperationsSetList[category_name].OPERATIONS):
  # for i, operation in enumerate(sorted(operations, key = lambda o: o.__name__)):
    # print(f"operations.{operation.__name__},")
    print(
      latexify(short_name(operation)),
      "&", 
      ", ".join([latexify(short_name(o)) for o in operation.input_program_types]),
      "&",
      latexify(short_name(operation.output_type)) if operation.output_type else "fn(input type)",
      "\\\\ \hline")
    # print(
    #   f"{i + 1}. {operation.__name__}".ljust(55), 
    #   [o.__name__ for o in operation.input_program_types], 
    #   " → ",
    #   operation.output_type.__name__ if operation.output_type else "<custom>")
    # print("\t", operation.might_have_bounded_minimum, operation.creates_gradients, operation.propagates_gradients)
