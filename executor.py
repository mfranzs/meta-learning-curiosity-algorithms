"""
Executes a program represented by a list of operations.
(Do not directly use these methods; instead use a Program class).
""" 

import pickle
import torch
from typing import List
import traceback

from mlca.program import Program
import mlca.operations as operations

class ProgramExecutionError(RuntimeError):
  pass

def _execute_program(
      program_operations: List[operations.Operation], input_values, data_structure_values, 
      optimizer_values, perform_mutations, print_on_error=True, profiler=None,
      i_episode=None):
  intermediate_values = {
    ** input_values, 
    ** data_structure_values, 
    ** optimizer_values
  }

  values_to_add_to_loss = []
  output_values = []

  for operation in program_operations:
    input_values = [intermediate_values[i] for i in operation.inputs]

    output_value = "UNSET"
    try: 
      output_value = operation.execute(
        input_values, profiler=profiler, i_episode=i_episode)
      output_values.append(output_value)
      intermediate_values[operation] = output_value

      if operation.add_to_loss:
        values_to_add_to_loss.append(operation.value_to_add_to_loss(
          input_values, output_value, profiler, i_episode
        ))

      if profiler is not None:
        profiler.tick(i_episode, str(type(operation)))
    except Exception as e:
      if (type(output_value) == str and output_value == "UNSET") or not operation.cached_output_type.is_correctly_formatted_value(output_value):
        if print_on_error or True:
          print("\n\n!!!!!!!!!!!!!!!!")
          print("Operation failed")
          print(operation)
          # print(input_values)
          print(e)
          traceback.print_exc()
          for inp in input_values:
            if inp is None:
              print("inp", inp)
            elif type(inp) is list:
              print("inp list", inp[0].shape)
            elif type(inp) is torch.Tensor:
              print("inp", inp.shape, inp.device)
            else:
              print("inp", inp)
          if output_value is None:
            print(type(operation), output_value)
          elif type(output_value) is list:
            print(type(operation), "list", output_value[0].shape)
          elif type(output_value) is torch.Tensor:
            print(type(operation), output_value.shape)
          else:
            print(type(operation), output_value)
          print("!!!!!!!!!!!!!!!!")
      raise ProgramExecutionError(e)

  for operation, output_value in zip(program_operations, output_values):
    try:
      assert operation.cached_output_type.value_class == type(output_value), (
        "wanted", operation.cached_output_type.value_class, "got", type(output_value), operation)
      assert operation.cached_output_type.is_correctly_formatted_value(output_value)
      assert operation.cached_output_type.is_valid_value(
          output_value)
    except Exception as e:
      if (type(output_value) == str and output_value == "UNSET") or not operation.cached_output_type.is_correctly_formatted_value(output_value):
        if print_on_error or True:
          print("\n\n!!!!!!!!!!!!!!!!")
          print("Operation failed")
          print(operation)
          # print(input_values)
          print(e)
          traceback.print_exc()
          for inp in input_values:
            if inp is None:
              print("inp", inp)
            elif type(inp) is list:
              print("inp list", inp[0].shape)
            elif type(inp) is torch.Tensor:
              print("inp", inp.shape, inp.device)
            else:
              print("inp", inp)
          if output_value is None:
            print(type(operation), output_value)
          elif type(output_value) is list:
            print(type(operation), "list", output_value[0].shape)
          elif type(output_value) is torch.Tensor:
            print(type(operation), output_value.shape)
          else:
            print(type(operation), output_value)
          print("!!!!!!!!!!!!!!!!")
      raise ProgramExecutionError(e)

  if perform_mutations:
    try:
      if len(values_to_add_to_loss) > 0:
        assert len(optimizer_values) == 1, f"Wrong # of optimizers given {optimizers}"
        optimizer = list(optimizer_values.values())[0]
        for v in values_to_add_to_loss:
          assert v.shape == tuple() or len(v.shape) == 1, v.shape
        loss = torch.sum(torch.stack([v.mean() for v in values_to_add_to_loss]))
        if profiler is not None:
            profiler.tick(i_episode, "Executor MinimizeValue: setup")

        optimizer.zero_grad()
        if profiler is not None:
            profiler.tick(i_episode, "Executor MinimizeValue: zero grad")
        loss.backward()
        if profiler is not None:
            profiler.tick(i_episode, "Executor MinimizeValue: backward")
        # for param in self.q_net.parameters():
        #     if param.grad is not None:
        #         param.grad.data.clamp_(-1, 1)
        optimizer.step()
        if profiler is not None:
            profiler.tick(i_episode, "Executor MinimizeValue: step")
    except Exception as e:
      if print_on_error or True:
        print("\n\n!!!!!!!!!!!!!!!!")
        print("Backprop gradients failed")
        for p in program_operations:
          print(p)
        print(values_to_add_to_loss)
        print("!!!!!!!!!!!!!!!!")
      raise ProgramExecutionError(e)

  return intermediate_values
