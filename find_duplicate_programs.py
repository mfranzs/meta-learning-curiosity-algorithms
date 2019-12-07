"""
Generate a "signature" for a program by simulating its outputs for 
tens of steps on random (but seeded) inputs. Compare these signatures 
to detect duplicate programs.
"""

import pickle
import random

import torch

import mlca.helpers.config
import mlca.helpers.util
from mlca.helpers.nn import one_hot
from mlca.program import Program
from mlca.executor import ProgramExecutionError
from mlca.program_types import get_action_space_size, NeuralNetworkWeightsObservationToFeatureVector32
from mlca.test_synthesized_programs_experiments import TspExperimentList
import mlca.operations as operations

class FakePolicy:
  def __init__(self, test_env):
    self.cnn_weights = NeuralNetworkWeightsObservationToFeatureVector32().create_empty(
      test_env, self
    )
  def act(self, state, a, b):
    return self.cnn_weights(state)

def get_program_signature(
    program: Program, test_env):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  random_seed = 0

  mlca.helpers.util.set_random_seed(random_seed)

  fake_policy = FakePolicy(test_env)

  data_structure_values, optimizer_values = program.initialize_program_structures(
    test_env, fake_policy)

  rewards = []
  for i in range(4): #3):
    mlca.helpers.util.set_random_seed(random_seed + 10 * i)

    shape = (
      test_env.observation_space.shape[2], 
      test_env.observation_space.shape[0], 
      test_env.observation_space.shape[1])

    state = torch.rand(shape,
                      device=device).unsqueeze(0)
    next_state = torch.rand(
        shape, device=device).unsqueeze(0)
    action = torch.randint(0, get_action_space_size(test_env.action_space),
                          (1,), device=device).unsqueeze(0)
    action_one_hot = one_hot(
        random.randint(0, get_action_space_size(
            test_env.action_space) - 1), get_action_space_size(test_env.action_space)
    ).to(device).unsqueeze(0)

    extrinsic_reward = torch.rand(1, device=device).unsqueeze(0)
    normalized_timestep = torch.rand(1, device=device).unsqueeze(0)
    intrinsic_reward = torch.rand(1, device=device).unsqueeze(0)

    input_values = {
        "observation_image": state,
        "action_one_hot": action_one_hot,
        "new_observation_image":  next_state,
        "extrinsic_reward":  extrinsic_reward,
        "normalized_timestep":  normalized_timestep,
        "intrinsic_reward":  intrinsic_reward,
    }

    input_values_by_variable = {
        i: input_values[i.name] for i in program.input_variables
    }

    assert set(i.name for i in program.input_variables).issubset(set(input_values.keys())), \
        ("available values", set(input_values.keys()), "requested values", set(
            i.name for i in program.input_variables))

    try: 
      reward = program.execute(
          input_values_by_variable,
          data_structure_values,
          optimizer_values,
          print_on_error=False
      ).item()
      rewards.append(reward)
    except ProgramExecutionError as e:
        rewards.append(None)
    except Exception as e:
      print(program)
      raise e

  return tuple(rewards)

def main():
  exp_name = "2-28-15x15-ppo-5-rollouts-500-steps"
  programs_name = "programs_curiosity_7_v6"

  programs_file_name = "pickles/" + programs_name + '.pickle'
  with open(programs_file_name, 'rb') as f:
      programs, program_inputs, data_structures, optimizers = pickle.load(f)

  params = TspExperimentList[exp_name]

  with params:
    seen_programs = {}
      
    duplicates = 0
    for i, program in enumerate(programs):
      program_signature = get_program_signature(program, test_env)

      if not None in program_signature:

        if program_signature not in seen_programs:
          seen_programs[program_signature] = []
        else:
          duplicates += 1
          
        seen_programs[program_signature].append(program)

      if i % 100 == 0:
        print(i, duplicates)

    for signature in sorted(seen_programs):
        print(signature)
        print(seen_programs[signature][0])

if __name__ == "__main__":
    main()

