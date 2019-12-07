import numpy as np
import time
import torch
from mlca.helpers.config import DefaultDevice

# Tested in mlca/curiosity/program_synthesis/scripts/misc/test_knn_speed.py

class TorchKNN:
  def __init__(self, buffer_size, feature_size, num_neighbors):
    self.buffer = torch.zeros(buffer_size, feature_size, device=DefaultDevice.current())
    self.nearest_neighbors = torch.zeros(num_neighbors, feature_size, device=DefaultDevice.current())

    self.buffer_size = buffer_size
    self.feature_size = feature_size
    self.num_neighbors = num_neighbors

    self.num_points = 0
    self.buffer_pos = 0

  def add(self, x):
    assert x.shape[0] == self.feature_size

    self.buffer[self.buffer_pos] = x

    self.num_points += 1
    self.buffer_pos += 1
    if self.buffer_pos >= self.buffer_size:
      self.buffer_pos = 0

  def predict(self, x):
    if self.num_points == 0:
      return torch.rand(self.feature_size)
    else:
      distances = torch.norm(
        self.buffer[:min(self.num_points, self.buffer_size)] - x, dim=1)

      _, indices = torch.topk(
        distances, min(self.num_neighbors, self.num_points),
        largest=False, sorted=False)
      nearest = self.buffer[indices]
      prediction = torch.mean(nearest, dim=0)

      assert prediction.shape == (self.feature_size, )
      return prediction


class TorchKNNRegressor:
  def __init__(self, buffer_size, feature_size, num_neighbors):
    self.query_buffer = torch.zeros(buffer_size, feature_size)
    self.target_buffer = torch.zeros(buffer_size, feature_size)
    self.nearest_neighbors = torch.zeros(num_neighbors, feature_size)

    self.buffer_size = buffer_size
    self.feature_size = feature_size
    self.num_neighbors = num_neighbors

    self.num_points = 0
    self.buffer_pos = 0

  def add(self, query, target):
    assert query.shape[0] == self.feature_size
    assert target.shape[0] == self.feature_size

    self.query_buffer[self.buffer_pos] = query
    self.target_buffer[self.buffer_pos] = target

    self.num_points += 1
    self.buffer_pos += 1
    if self.buffer_pos >= self.buffer_size:
      self.buffer_pos = 0

  def predict(self, x):
    if self.num_points == 0:
      return torch.rand(self.feature_size)
    else:
      distances = torch.norm(
        self.query_buffer[:min(self.num_points, self.buffer_size)] - x, dim=1)

      _, indices = torch.topk(
        distances, min(self.num_neighbors, self.num_points),
        largest=False, sorted=False)
      nearest = self.target_buffer[indices]
      prediction = torch.mean(nearest, dim=0)

      assert prediction.shape == (self.feature_size, )
      return prediction
