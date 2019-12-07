import torch
import argparse
import pprint
from typing import List

def argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment_id", required=True,
                      help="id of the experiment we want to run; auto-selects the parameters")
  parser.add_argument("--render", action="store_true",
                      help="render the environment", default=False)
  parser.add_argument("--dont_train", action="store_true",
                      help="don't train the model", default=False)
  parser.add_argument("--dont_save", action="store_true",
                      help="don't save the model", default=False)
  parser.add_argument("--dont_load", action="store_true",
                      help="don't load the model", default=False)
  parser.add_argument("--cpu", action="store_true", default=False)
  parser.add_argument("--profiler", action="store_true",
                      help="print profiler data", default=False)
  return parser


def clean_experiment_id(experiment_id):
  # Only get the experiment_id before --version
  return experiment_id.split("--version")[0]


def get_params(experiments, experiment_id, print_params=True, recursing=False):
  print("WARNING: get_params is deprecated")
  experiment_id = clean_experiment_id(experiment_id)
  if experiment_id not in experiments:
    raise RuntimeWarning("The experiment ID "+experiment_id +
                         " does not exist! Valid ids: " + str(experiments.keys()))

  e = experiments[experiment_id]
  e["__EXPERIMENT_ID__"] = experiment_id

  if "__PARENT__" in e:
    parent = get_params(experiments, e["__PARENT__"], print_params, True)
    e = {**parent, **e}

  if not recursing and print_params:
    print("Parameters: ", experiment_id)
    pprint.pprint(e)

  assert e.get("BUGGED", None) is None, e.get("BUGGED")

  return e

def get_device_and_set_default(args = None):
  use_cuda = torch.cuda.is_available() and not (args is not None and args.cpu)

  device = "cuda" if use_cuda else "cpu"
  print("Device", device, " - Cuda available?", torch.cuda.is_available())
  if device == "cuda":
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
  else:
      torch.set_default_tensor_type(torch.FloatTensor)

  return device
  
class DefaultDevice:
  active_default_parameters: List[str] = []

  def __init__(self, default_device):
    self.default_device = default_device

  def __enter__(self):
    self.active_default_parameters.append(self.default_device)
    return self

  def __exit__(self, type, value, traceback):
    self.active_default_parameters.pop()

  @classmethod
  def current(cls):
    if len(cls.active_default_parameters) == 0:
      raise RuntimeError(f"No current global default device set. Use a with statement to set the default device.")
    return cls.active_default_parameters[-1]
  