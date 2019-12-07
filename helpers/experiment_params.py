"""
A small library that represents experiment parameters using 
Python Dataclasses. Supports creating lists of experiment parameters, 
chaining experiment paramters, and registering a current globally active 
parameter instance for a given paramter class (using a Python context).
"""

import dataclasses 
from collections import OrderedDict
import pprint
from typing import List, Optional

ExperimentId = str

class ExperimentParameters():
  active_parameters: Optional[List] = None

  @classmethod
  def current(cls):
    if cls.active_parameters is None or len(cls.active_parameters) == 0:
      raise RuntimeError(f"No current global parameters set for {cls.__name__}. Use a with statement to set the current context.")
    return cls.active_parameters[-1]

  @classmethod
  def _set_active_parameters(cls, active_parameters):
    # Make the active parameter list when needed so we have a different one
    # for every parameters class
    if cls.active_parameters is None:
      cls.active_parameters = []

    cls.active_parameters.append(active_parameters)

  @classmethod
  def _clear_active_parameters(cls):
    cls.active_parameters.pop()

  def register_experiment_id(self, _experiment_id: ExperimentId):
    assert not hasattr(self, '_experiment_id'), "Warning: You cannot use the reserved field _experiment_id in your experiment params."
    self._experiment_id = _experiment_id

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def __enter__(self):
    self._set_active_parameters(self)
    return self

  def __exit__(self, type, value, traceback):
    self._clear_active_parameters()

class ExperimentParameterList(dict):
  def __setitem__(self, key: ExperimentId, item: ExperimentParameters):
    assert key not in self.__dict__, f"The experiment {key} has already been registered."
    self.__dict__[key] = item
    item.register_experiment_id(key)

  def get(self, key: ExperimentId, print_params=True):
    if print_params:
      print("Get experiments", key)
      print(pprint.pformat(self.__dict__[key]))
    return self[key]

  def __getitem__(self, key: ExperimentId):
    return self.__dict__[key]
  