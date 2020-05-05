import time
import math
from dataclasses import dataclass
from typing import Any

import mlca.helpers.debug

import colored_traceback
colored_traceback.add_hook()

class Profiler:
    def __init__(self, logger, print_enabled=True):
        self.logger = logger
        self.print_enabled = print_enabled
        self.reset(-1, False)

    def print(self, *s):
        if self.print_enabled:
            print(*s)

    def reset(self, i_episode, log=True):
        if log:
            self.tick(i_episode, "reset")
            self.print(
                "--------------------------- Time since last reset: ",
                (time.time() - self.last_reset),
            )

        self.s = time.time()
        self.last_reset = time.time()

    def tick(self, i_episode, name):
        if self.logger and i_episode:
            self.logger.add_scalar(
                "Profiler: " + name, time.time() - self.s, i_episode,
            )

        self.print("---", name, time.time() - self.s)
        self.s = time.time()

class Logger():
  def __init__(self, tensorboard_writer):
    self.tensorboard_writer = tensorboard_writer
    
    # Data for averaging values on a single i_episode
    self.current_i_episode = {}
    self.current_sum = {}
    self.current_count = {}

  def add_scalar(self, name, value, i_episode, average=False):
    if self.tensorboard_writer is not None:
      if average:
        # Average values on the current timestep. While on that timestep, keep updating
        # our counter. If the timestep changes, compute and save the average. When the 
        # logger is destroyed, do the same in __del__.
        # This reduces the amount we're spamming tensorboard, and also gives us nicer numbers.
        if self.current_i_episode.get(name, -1) == i_episode:
          self.current_sum[name] += value
          self.current_count[name] += 1
        else:
          assert self.current_i_episode.get(name, -1) < i_episode, \
              f"You're logging {name} on an earlier episode than the current one! {value} {i_episode} {self.current_i_episode.get(name, None)}"

          if self.current_count.get(name, 0) > 0: 
            self.tensorboard_writer.add_scalar(
              name, 
              self.current_sum[name] / self.current_count[name],
              self.current_i_episode[name])

          self.current_i_episode[name] = i_episode
          self.current_sum[name] = value
          self.current_count[name] = 1

      else:
        assert self.current_i_episode.get(name, -1) < i_episode, \
            f"You're logging {name} multiple times per episode! {value} {i_episode} {self.current_i_episode.get(name, None)}"

        self.tensorboard_writer.add_scalar(name, value, i_episode)
        self.current_i_episode[name] = i_episode

  def __del__(self):
    # Store the averaged counts from the last timestep
    for name in self.current_count:
      if self.current_count[name] > 0:
        self.tensorboard_writer.add_scalar(
          name,
          self.current_sum[name] / self.current_count[name],
          self.current_i_episode[name])

@dataclass
class InMemoryLoggerAverageStats:
  # TODO
  name: Any
  sums: Any
  counts: Any

class InMemoryLogger():
  # Note this does something totally different than Logger
  def __init__(self):
    self.sums = {}
    self.counts = {}

  def add_scalar(self, name, value, i_episode):
    if name not in self.sums:
      self.sums[name] = 0
      self.counts[name] = 0
    self.sums[name] += value
    self.counts[name] += 1


  def print_avg_stats(self):
    self._print_avg_stats(self.counts, self.sums)

  @staticmethod
  def _print_avg_stats(counts, sums):
    print("------")
    if len(counts) == 0:
      print("No stats to print")
      return 
    min_count = min(counts.values())
    for name in counts:
      avg_time = str(sums[name] / counts[name]).ljust(30)
      times = str(math.ceil(counts[name] / min_count)).ljust(6)
      total_time = sums[name] / min_count
      print(f"{name.ljust(70)}: {avg_time}  x{times}  {total_time}")

  @staticmethod
  def combine_avg_stats(counts_list, sums_list):
    counts = {}
    sums = {}
    for c, s in zip(counts_list, sums_list):
      for name in c:
        if name not in sums:
          sums[name] = 0
          counts[name] = 0
        sums[name] += s[name]
        counts[name] += c[name]
    return counts, sums

  def avg_stats(self) -> InMemoryLoggerAverageStats:
    return InMemoryLoggerAverageStats({
      name: self.sums[name] / self.counts[name]
      for name 
      in self.sums
    }, self.sums, self.counts)
