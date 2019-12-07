import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
from tqdm import tqdm
import time
import random
import os
import gzip
import math
from typing import *

import mlca.helpers.debug
from mlca.program import Program
from mlca.test_synthesized_programs_experiments import TspExperimentList, TspParams
import mlca.helpers.config
from mlca.helpers.plotting import scatter_vertical_histogram
from mlca.search_program_experiments import SearchExperimentList

ProgramData = collections.namedtuple('ProgramData', [
  'index', 'curiosity_program', 'reward_combiner_program', 'results', 'stats'])

def main():
  experiment_id = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"

  params = TspExperimentList[experiment_id]
  with params: 
    simulator_params = SearchExperimentList[TspParams.current().SEARCH_PROGRAMS_EXPERIMENT_ID]

    print(TspParams.current().EXPERIMENT_TYPE)
    if TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.CURIOSITY_SEARCH: 
      data, curiosity_programs_with_results, \
            curiosity_programs, curiosity_program_inputs, \
            curiosity_data_structures, curiosity_optimizers, \
            reward_combiner_programs, reward_combiner_program_inputs, \
            reward_combiner_data_structures, reward_combiner_optimizers, program_results_data \
        = load_curiosity_program_data(
          TspParams.current().CURIOSITY_PROGRAMS_NAME, 
          TspParams.current().REWARD_COMBINER_PROGRAMS_NAME, 
          experiment_id,
          TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID)
    elif TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.REWARD_COMBINER_SEARCH:
      data, curiosity_programs_with_results, \
          curiosity_programs, curiosity_program_inputs, \
          curiosity_data_structures, curiosity_optimizers, \
          reward_combiner_programs, reward_combiner_program_inputs, \
          reward_combiner_data_structures, reward_combiner_optimizers, program_results_data \
          = load_reward_combiner_program_data(
              TspParams.current().CURIOSITY_PROGRAMS_NAME,
              TspParams.current().REWARD_COMBINER_PROGRAMS_NAME,
              experiment_id, 
              TspParams.current().FIXED_CURIOSITY_PROGRAM_ID)

    # print(data[0].results.trials_states)

    print("Done loading")

    programs_with_error = [p for p in data if p.results.execution_had_error]

    # for p in programs_with_error:
    #   print(p.results.execution_had_error)
    #   print(str(p.results.error))
    #   print_program_data(p, experiment_id)

    # print(place_programs_in_buckets(data))

    # print("Num programs", len(programs))
    # print("Num evaluated programs", len(program_results_data))
    # print("Num programs with error", len(programs_with_error))

    # _view_programs(curiosity_programs)

    # _throughput(data)

    stdevs = np.array([p.stats["mean_performance_std"] for p in data if p.stats])
    means = np.array([p.stats["mean_performance"] for p in data if p.stats])
    print("Data points", len(data))
    print("means", len(means))
    print(
      "n", len(stdevs), 
      "avg std", stdevs.mean(),
      "mean performance", means.mean(),
      ".25 quantile", np.quantile(means, 0.25),
      ".75 quantile", np.quantile(means, 0.75),
      "total CPU time", sum([p.results.elapsed_time for p in data]), 
      "# w. error", len(programs_with_error),
      "# w. had_early_termination_data", len([d for d in data if d.results.had_early_termination_data])
    )

    print("total amount of steps", 
      sum(_total_steps(p.results.trials_rollouts_episode_lengths) for p in data if p.stats),
      "max amount of steps",
      TspParams.current().STEPS_PER_ROLLOUT * TspParams.current().NUM_ROLLOUTS_PER_TRIAL * \
      TspParams.current().NUM_TRIALS_PER_PROGRAM *  len([p for p in data if p.stats])
    )

    # print_profiler(data)
    # plot_scatter_of_every_trials(data, params)
    # plot_scatter_of_every_trial_best_performance(data, params)
    # plot_scatter_of_every_trial_last_performance(data, params)
    # plot_scatter_of_every_trial_mean_performance(data, params)
    
    # plot_scatter_of_mean_plus_minus_std(data, experiment_id)
    # plot_scatter_of_max_plus_minus_std(data, experiment_id)
    # print(ids_of_best_n_programs(data, 16))
    # quit()
    
    # from mlca.simulate_search import _plot_program_evaluations
    # _plot_program_evaluations(data, None, simulator_params, params)

    # plot_histogram_of_runtime(data)
    # plot_histogram_of_average_steps_to_reach_n_unique_states(data)
    # # compare_mean_with_avg_steps_to_reach_n_unique_states(data, 10)
    # plot_histogram_of_episode_length(data)
    # quit() 

    # plot_histogram_of_mean_performances(data)
    #analyze_errors(
    #  programs_with_error, data, experiment_id)

    # plot_scatter_of_selection_index_vs_total_timesteps(data)

    # plot_violin_of_mean_performances(data)
    # plot_scatter_of_stdev_vs_mean(data)
    # plot_histogram_of_stdevs(data)

    # print(program_inputs[1])
    # for p in data:
    #   # if p.stats["mean_performance"] > 60:
    #   if program_inputs[1] in p.program.update_program[-1].input_set:
    #     print(p)

    # print("")
    # print("Smallest program above threshold")
    # for p in smallest_programs_above_threshold(data, 80):
    #   print_program_data(p)

    # print("Best programs"
    best = programs_by_mean_performance(data)
    print("REMOVE THIS HACK")
    best = best[-16:]
    print(len(best))
    print([d.curiosity_program.program_id for d in best])
    for i in range(13):
      print("------")
      print(i, "from top")
      print("------")
      d = best[-i - 1]
      # worst: # d = programs_by_mean_performance(data)[i]
      print_program_data(d, experiment_id)
      d.curiosity_program.visualize_as_graph(i)

      if d.reward_combiner_program is not None:
        d.reward_combiner_program.visualize_as_graph(str(i) + "combiner")
  
def ids_of_best_n_programs(data, n):
  sorted_data_mean = sorted([d for d in data if d.stats],
                       key=lambda d: d.stats["mean_performance"]
                    )
  best_data_mean = sorted_data_mean[-n:]

  # sorted_data_mean_std = sorted([d for d in data if d.stats],
  #                      key=lambda d: d.stats["mean_performance"] - \
  #                        d.stats["mean_performance_std"]
  #                   )
  # best_data_mean_std = sorted_data_mean_std[-n:]

  # plot_scatter_of_mean_plus_minus_std(best_data_mean)
  # plot_scatter_of_mean_plus_minus_std(best_data_mean_std)

  for i, d in enumerate(best_data_mean):
    print(d.curiosity_program.program_id, d.stats)
    d.curiosity_program.visualize_as_graph(i)

  return [d.curiosity_program.program_id for d in best_data_mean]

def load_program_data(experiment_id, filter_trials=None):
  params = TspExperimentList[experiment_id]

  if params.EXPERIMENT_TYPE == TspParams.ExperimentType.CURIOSITY_SEARCH:
    return load_curiosity_program_data(
            params.CURIOSITY_PROGRAMS_NAME,
            params.REWARD_COMBINER_PROGRAMS_NAME,
            experiment_id,
            params.FIXED_REWARD_COMBINER_PROGRAM_ID,
            filter_trials)
  elif params.EXPERIMENT_TYPE == TspParams.ExperimentType.REWARD_COMBINER_SEARCH:
    return load_reward_combiner_program_data(
        params.CURIOSITY_PROGRAMS_NAME,
        params.REWARD_COMBINER_PROGRAMS_NAME,
        experiment_id,
        params.FIXED_CURIOSITY_PROGRAM_ID,
        filter_trials)

def load_curiosity_program_data(
    curiosity_programs_name, reward_combiner_programs_name,
    experiment_id,
    fixed_reward_combiner_program_id, filter_trials=None) -> List[ProgramData]:

  params = TspExperimentList[experiment_id]

  curiosity_programs_file = "pickles/" + curiosity_programs_name + '.pickle'

  evaluation_file = f"pickles/{experiment_id}_evaluations.pickle"
  evaluation_folder = f"pickles/{experiment_id}_evaluations/"

  print("Loading programs", curiosity_programs_file)
  with open(curiosity_programs_file, 'rb') as f:
        curiosity_programs, curiosity_program_inputs, curiosity_data_structures, curiosity_optimizers = pickle.load(f)
  print("# curiosity_programs", len(curiosity_programs))
  
  if reward_combiner_programs_name is not None:
    reward_combiner_programs_file = "pickles/" + reward_combiner_programs_name + '.pickle'
    with open(reward_combiner_programs_file, 'rb') as f:
        reward_combiner_programs, reward_combiner_program_inputs, reward_combiner_data_structures, reward_combiner_optimizers = pickle.load(
            f)
    print("# curiosity_programs", len(reward_combiner_programs))
  else: 
    reward_combiner_programs, reward_combiner_program_inputs, reward_combiner_data_structures, reward_combiner_optimizers = None, None, None, None

  print("Done loading programs")

  if os.path.exists(evaluation_file):
    quit("Loading evaluations from a file is deprecated")
  else:
    r = []
    curiosity_programs_with_results = []
    program_results_data = []
    
    print("curiosity_programs", len(curiosity_programs))
    print("experiment_id", experiment_id)
    print("reward_combiner_programs_name", reward_combiner_programs_name)
    print("fixed_reward_combiner_program_id", fixed_reward_combiner_program_id)
    program_results_filenames = set(os.listdir(evaluation_folder) if os.path.exists(evaluation_folder) else [])

    for index, curiosity_program in tqdm(enumerate(curiosity_programs), "Loading existing evaluation data", total=len(curiosity_programs)):
      if reward_combiner_programs_name is None and experiment_id != "2-57-15x15-ppo-5-rollouts-500-steps-four-rooms":
        # Backwards-compatibility w. before had reward_combiner
        program_pickle_results_file = f"{index}_None.pickle.gz"
      else: 
        program_pickle_results_file = f"{index}_{fixed_reward_combiner_program_id}.pickle.gz"

      if program_pickle_results_file in program_results_filenames: 
        with gzip.open(evaluation_folder + program_pickle_results_file, 'rb') as results_file:
          serialized_obj = results_file.read()
          results = pickle.loads(serialized_obj)
        program_results_data.append(results)

        if type(curiosity_program) is Program:
          curiosity_program = Program(
            curiosity_program.forward_program, curiosity_program.update_program, 
            curiosity_program_inputs, curiosity_data_structures, curiosity_optimizers,
            index)

        if results.curiosity_program_id is not None:
          assert results.curiosity_program_id == index

        curiosity_programs_with_results.append(curiosity_program)
        r.append(
            ProgramData(index, curiosity_program, None, results, # TODO: Load reward program
                        _stats_for_program(results, filter_trials)))

    # assert len(program_results_filenames) == len(r), (len(program_results_filenames), len(r))

    return r, curiosity_programs_with_results, \
        curiosity_programs, curiosity_program_inputs, \
        curiosity_data_structures, curiosity_optimizers, \
        reward_combiner_programs, reward_combiner_program_inputs, \
        reward_combiner_data_structures, reward_combiner_optimizers, \
        program_results_data


def load_reward_combiner_program_data(
        curiosity_programs_name, reward_combiner_programs_name, experiment_id, \
        fixed_curiosity_program_id, filter_trials=None):
  params = mlca.helpers.config.get_params(
      experiments, experiment_id)
      
  curiosity_programs_file = "pickles/" + curiosity_programs_name + '.pickle'
  reward_combiner_programs_file = "pickles/" + \
      reward_combiner_programs_name + '.pickle'

  evaluation_file = f"pickles/{experiment_id}_evaluations.pickle"
  evaluation_folder = f"pickles/{experiment_id}_evaluations/"

  print("Loading programs")
  with open(curiosity_programs_file, 'rb') as f:
    curiosity_programs, curiosity_program_inputs, curiosity_data_structures, curiosity_optimizers = pickle.load(
        f)
    fixed_curiosity_program = curiosity_programs[fixed_curiosity_program_id]
    # assert fixed_curiosity_program.program_id == fixed_curiosity_program_id
  print("# curiosity_programs", len(curiosity_programs))

  with open(reward_combiner_programs_file, 'rb') as f:
      reward_combiner_programs, reward_combiner_program_inputs, reward_combiner_data_structures, reward_combiner_optimizers = pickle.load(
          f)
  print("# reward_combiner_programs", len(reward_combiner_programs))

  print("Done loading programs")

  if os.path.exists(evaluation_file):
    quit("Loading evaluations from a file is deprecated")
  else:
    r = []
    reward_combiner_programs_with_results = []
    program_results_data = []

    program_results_filenames = set(os.listdir(evaluation_folder))

    for index, reward_combiner_program in tqdm(enumerate(reward_combiner_programs), "Loading existing evaluation data"):
      program_pickle_results_file = f"{fixed_curiosity_program_id}_{reward_combiner_program.program_id}.pickle.gz"
      if program_pickle_results_file in program_results_filenames:
        with gzip.open(evaluation_folder + program_pickle_results_file, 'rb') as results_file:
          serialized_obj = results_file.read()
          results = pickle.loads(serialized_obj)
          # print(results)
          # if len(reward_combiner_programs_with_results) == 10:
          #   break

        if type(reward_combiner_program) is Program:
          reward_combiner_program = Program(
              reward_combiner_program.forward_program, reward_combiner_program.update_program, 
              reward_combiner_program_inputs, reward_combiner_data_structures, reward_combiner_optimizers, 
              index)

        program_results_data.append(results)
        reward_combiner_programs_with_results.append(reward_combiner_program)
        r.append(
            ProgramData(index, fixed_curiosity_program, reward_combiner_program, results,
                        _stats_for_program(results, filter_trials)))

    return r, reward_combiner_programs_with_results, \
        curiosity_programs, curiosity_program_inputs, \
        curiosity_data_structures, curiosity_optimizers, \
        reward_combiner_programs, reward_combiner_program_inputs, \
        reward_combiner_data_structures, reward_combiner_optimizers, program_results_data

# Toothpaste plot for MEAN
def plot_scatter_of_mean_plus_minus_std(data: ProgramData, experiment_id):
  data = [d for d in data if not d.results.early_terminated]
  # for i, d in enumerate(sorted(data, key=lambda p: p.results.start_time)):
  #   print((i, d.results.selected_index))
  # assert i == d.results.selected_index"], (i, d.results["selected_index)
  # d.results.selected_index = i

  # and d.results.get("selected_index", None) is not None and not d.results.early_terminated]
  data = [d for d in data if d.stats]
  print("Data w. index", len((data)))

  # def m(p): return (p.stats["mean_performance"]
  #                   if not p.results.early_terminated else math.nan)
  def m(p): return p.stats["mean_performance"]

  data = sorted(data, key=lambda p: p.stats["mean_performance"])
  means = [m(p) for p in data if p.stats]

  bottom = [m(p) - p.stats["mean_performance_std"]
            for p in data if p.stats]
  top = [m(p) + p.stats["mean_performance_std"]
         for p in data if p.stats]

  scatter_vertical_histogram(
    range(len(means)),
    means,
    binwidth=10
  )
  # plt.scatter(
  #     range(len(means)),
  #     means,
  #     s=1,
  #     c="black")
  # plt.scatter(
  #     range(len(means)),
  #     top,
  #     s=1,
  #     c="green"
  # )
  # plt.scatter(
  #     range(len(means)),
  #     bottom,
  #     s=1,
  #     c="red"
  # )
  plt.show()


# Toothpaste plot for MAX
def plot_scatter_of_max_plus_minus_std(data: ProgramData, experiment_id):
  data = [d for d in data if not d.results.early_terminated]
  # for i, d in enumerate(sorted(data, key=lambda p: p.results.start_time)):
  #   print((i, d.results.selected_index))
  # assert i == d.results.selected_index"], (i, d.results["selected_index)
  # d.results.selected_index = i

  # and d.results.get("selected_index", None) is not None and not d.results.early_terminated]
  data = [d for d in data if d.stats]
  print("Data w. index", len((data)))

  # def m(p): return (p.stats["mean_performance"]
  #                   if not p.results.early_terminated else math.nan)
  def m(p): return p.stats["trial_mean(rollout_mean(max_episode_in_rollout))_performance"]

  data = sorted(data, key=m)
  means = [m(p) for p in data if p.stats]

  bottom = [m(p) - p.stats["trial_mean(rollout_mean(max_episode_in_rollout))_performance_std"]
            for p in data if p.stats]
  top = [m(p) + p.stats["trial_mean(rollout_mean(max_episode_in_rollout))_performance_std"]
         for p in data if p.stats]

  plt.scatter(
      range(len(means)),
      means,
      s=1,
      c="black")
  plt.scatter(
      range(len(means)),
      top,
      s=1,
      c="green"
  )
  plt.scatter(
      range(len(means)),
      bottom,
      s=1,
      c="red"
  )
  plt.xlabel("Program index, sorted by trial_mean(rollout_mean(max_episode_in_rollout))_performance")
  plt.ylabel("trial_mean(rollout_mean(max_episode_in_rollout))_performance")
  plt.title(experiment_id)
  plt.show()

def plot_scatter_of_every_trials(data: ProgramData, params):
  data = [d for d in data if not d.results.early_terminated]
  data = [d for d in data if d.stats] 
  data = sorted(data, key=lambda p: p.stats["mean_performance"])

  for trial in range(TspParams.current().NUM_TRIALS_PER_PROGRAM):
    trial_perf = [np.array(d.results.trials_rollouts_mean_reward[trial]).mean() for d in data]
    plt.scatter(
        range(len(data)),
        trial_perf,
        s=1, 
        c="black")

  plt.title(TspParams.current().__EXPERIMENT_ID__)
  plt.xlabel("Program index (by mean performance)")
  plt.ylabel("Mean performance of finished episodes, per trial")
  plt.show()

def plot_scatter_of_every_trial_last_performance(data: ProgramData, params):
  data = [d for d in data if not d.results.early_terminated]
  data = [d for d in data if d.stats] 
  data = sorted(data, key=lambda p: p.stats["mean_performance"])

  print("Data w. index", len((data)))

  for trial in range(TspParams.current().NUM_TRIALS_PER_PROGRAM):
    trial_perf = [_mean_last_episode_performance(d, trial) for d in data]
    plt.scatter(
        range(len(data)),
        trial_perf,
        s=1, 
        c="black")

  # plt.ylim(top=200)
  plt.title(TspParams.current().__EXPERIMENT_ID__)
  plt.xlabel("Program index (by mean performance)")
  plt.ylabel("Performance of last episode, per trial")
  plt.show()

def plot_scatter_of_every_trial_mean_performance(data: ProgramData, params):
  data = [d for d in data if not d.results.early_terminated]
  data = [d for d in data if d.stats]
  data = sorted(data, key=lambda p: p.stats["mean_performance"])

  print("Data w. index", len((data)))

  for trial in range(TspParams.current().NUM_TRIALS_PER_PROGRAM):
    trial_perf = [
       np.array([np.array(r).mean() for r in d.results.trials_rollouts_episode_end_rewards[trial]]).mean()
       for d in data]
    plt.scatter(
        range(len(data)),
        trial_perf,
        s=1,
        c="black")

  # plt.ylim(top=200)
  plt.title(TspParams.current().__EXPERIMENT_ID__)
  plt.xlabel("Program index (by mean performance)")
  plt.ylabel("Mean episode performance, per trial")
  plt.show()

def plot_scatter_of_every_trial_best_performance(data: ProgramData, params):
  data = [d for d in data if not d.results.early_terminated]
  data = [d for d in data if d.stats]
  data = sorted(data, key=lambda p: p.stats["mean_performance"])

  print("Data w. index", len((data)))

  for trial in range(TspParams.current().NUM_TRIALS_PER_PROGRAM):
    trial_perf = [_best_episode_performance(d, trial) for d in data]
    plt.scatter(
        range(len(data)),
        trial_perf,
        s=1,
        c="black")

  plt.title(TspParams.current().__EXPERIMENT_ID__)
  plt.xlabel("Program index (by mean performance)")
  plt.ylabel("Performance of best episode, per trial")
  plt.show()

def _mean_last_episode_performance(d, trial):
  return np.array(
    [r[-1] for r in d.results.trials_rollouts_episode_end_rewards[trial]]).mean()

def _best_episode_performance(d, trial):
  return np.array([np.array(r).max() for r in d.results.trials_rollouts_episode_end_rewards[trial]]).mean()

def _avg_steps_to_reach_n_unique_states(p: ProgramData, n_unique_states=50):
  num_steps = []

  for trial_data in p.results.trials_states:
    for rollout_states in trial_data:
      states_reached = set()

      for step, state in enumerate(rollout_states):
        states_reached.add(state)
        if len(states_reached) >= n_unique_states:
          break

      num_steps.append(step)

  return np.array(num_steps).mean()

def plot_histogram_of_mean_performances(data):
  means = [ p.stats["mean_performance"] for p in data if p.stats]

  # print(data[0])
  print(len(means))
  print(sorted(means))

  n, bins, patches = plt.hist(means, 50, facecolor='g', alpha=0.75)

  plt.xlabel('Agent Performance')
  plt.ylabel('Agent Count')
  plt.title('Distribution of Agent Performance with Different Programs')
  plt.grid(True)
  plt.show()


def plot_violin_of_mean_performances(data):
  data = [p for p in data if p.stats]
  sizes = list(sorted(set(
    len(p.program.forward_program + p.program.update_program) for p in data
  )))
  means_per_size = [
    [p.stats["mean_performance"] for p in data if \
      len(p.program.forward_program + p.program.update_program) == size]
    for size in sizes]

  plt.violinplot(means_per_size, sizes, points=80, vert=False, widths=0.7,
                        showmeans=True, showextrema=True, showmedians=True)

  plt.xlabel('# States Reached')
  plt.ylabel('Program Size')
  plt.title('Distribution of # States Reached vs Program Size')
  plt.grid(True)
  plt.show()

def plot_histogram_of_episode_length(data):
  l = []
  num_episodes = []
  for p in data:
    if p.stats is not None:
      for t in p.results.trials_rollouts_episode_lengths:
        for r in t:
          num_episodes.append(len(r))
          for e in r:
            l.append(e)
            
  print("Avg episode len", np.array(l).mean(), "Avg # episodes (per rollout)", np.array(num_episodes).mean())

  n, bins, patches = plt.hist(
          l, 50, density=True, facecolor='g', alpha=0.75)

  plt.xlabel('Episode Length')
  plt.ylabel('Probability')
  plt.grid(True)
  plt.show()

  n, bins, patches = plt.hist(
    num_episodes, 50, density=True, facecolor='g', alpha=0.75)

  plt.xlabel('Num Episodes Per Rollout')
  plt.ylabel('Probability')
  plt.grid(True)
  plt.show()
            
  
def plot_histogram_of_runtime(data):
  stdevs = [p.results.elapsed_time for p in data if p.stats]

  n, bins, patches = plt.hist(
      stdevs, 50, density=True, facecolor='g', alpha=0.75)

  plt.xlabel('Time')
  plt.ylabel('Probability')
  plt.grid(True)
  plt.show()

def compare_mean_with_avg_steps_to_reach_n_unique_states(data, n_unique_states=50):
  avg_steps = [_avg_steps_to_reach_n_unique_states(p, n_unique_states) for p in data if p.stats]
  means = [p.stats["mean_performance"] for p in data if p.stats]

  plt.scatter(means, avg_steps, alpha=.1)

  plt.xlabel('Avg # of States Reached')
  plt.ylabel(f'Avg Steps to Reach {n_unique_states} Unique States')
  plt.show()

def plot_histogram_of_average_steps_to_reach_n_unique_states(data, n_unique_states=50):
  avg_steps = [_avg_steps_to_reach_n_unique_states(p, n_unique_states) for p in data if p.stats]

  n, bins, patches = plt.hist(
      avg_steps, 50, density=True, facecolor='g', alpha=0.75)

  plt.xlabel(f'Average Steps to Reach {n_unique_states} Unique States')
  plt.ylabel('Probability')
  plt.grid(True)
  plt.show()

def plot_histogram_of_stdevs(data):
  stdevs = [p.stats["mean_performance_std"] for p in data if p.stats]

  n, bins, patches = plt.hist(
      stdevs, 50, density=True, facecolor='g', alpha=0.75)

  plt.xlabel('Std # of States Reached')
  plt.ylabel('Probability')
  plt.title('Distribution of Std')
  plt.grid(True)
  plt.show()

def plot_scatter_of_selection_index_vs_total_timesteps(data):
  means = [p.results.selected_index for p in data if p.stats]
  times = [_total_steps(p.results.trials_rollouts_episode_lengths)
            for p in data if p.stats]

  plt.scatter(means, times, alpha=.1)

  print(max(times))

  plt.xlabel('Selection Index')
  plt.ylabel('Total Timesteps')
  plt.show()

def plot_scatter_of_stdev_vs_mean(data):
  means = [p.stats["mean_performance"] for p in data if p.stats]
  stdevs = [p.stats["mean_performance_std"] for p in data if p.stats]
  
  plt.scatter(means, stdevs, alpha=.1)

  plt.xlabel('Avg # of States Reached')
  plt.ylabel('Stdev # of States Reached')
  plt.show()

def programs_by_mean_performance(data):
  return sorted([d for d in data if d.stats], 
                key=lambda d: d.stats["mean_performance"]
  )

# for program, program_result in zip(programs, program_results_data):
#   if program_result["execution_had_error"]:
#     print(program)
#     print(program_result)


def _stats_for_program(results, filter_trials = None):
  if results is None:
    return None
  elif results.execution_had_error:
    return None
  elif results.trials_rollouts_mean_reward == None:
    return results.states_reached
  else:
    # t = results.trials_rollouts_episode_end_rewards
    # trial_means = np.array([
    #   np.array([x[-1] for x in r]).max() for r in t])
    t = results.trials_rollouts_mean_reward
    trial_means = np.array([np.array(x).mean() for x in t])
    trial_maxes = np.array([np.array([np.array(r).max() for r in t]).mean() for t in results.trials_rollouts_episode_end_rewards])
    trial_lasts = np.array([np.array([r[-1] for r in t]).mean() for t in results.trials_rollouts_episode_end_rewards])

    if filter_trials:
      trial_means = filter_trials(trial_means)
    # print(trial_means, trial_means.std(), len(trial_means))
    return {
      "mean_performance": trial_means.mean(),
      "mean_performance_std": trial_means.std() / math.sqrt(len(trial_means)),
      "trial_mean(rollout_mean(max_episode_in_rollout))_performance": trial_maxes.mean(),
      "trial_mean(rollout_mean(max_episode_in_rollout))_performance_std": trial_maxes.std() / math.sqrt(len(trial_maxes)),
      "trial_mean(rollout_mean(last_episode_in_rollout))_performance": trial_lasts.mean(),
      "trial_mean(rollout_mean(last_episode_in_rollout))_performance_std": trial_lasts.std() / math.sqrt(len(trial_lasts)),
    }

def _throughput(data):
  if len(data) > 0:
    average_elapsed_time = sum(
      p.results.elapsed_time for p in data if not p.resultsexecution_had_error) / len(data)
    NUM_IN_PARALLEL = 8 * 8
    print("average_elapsed_time", average_elapsed_time,
          average_elapsed_time / NUM_IN_PARALLEL, NUM_IN_PARALLEL)
  else:
    print("No data")

def print_profiler(data):
  data = [d for d in data if not d.results.execution_had_error]
  # for d in data:
  #   if not d.results.execution_had_error:
  #     avg_stats, sums, counts = d.results.avg_stats
  #     print("---------------------")
  counts, sums = mlca.helpers.debug.InMemoryLogger.combine_avg_stats(
    [d.results.avg_stats[2] for d in data],
    [d.results.avg_stats[1] for d in data]
  )
  mlca.helpers.debug.InMemoryLogger._print_avg_stats(counts, sums)


def analyze_errors(programs_with_error, programs, experiment_id):
  #for p in programs_with_error:
  #  print("----------------------")
  #  print(p.curiosity_program)
  #  print(p.results.error)

  wasted_time = sum(p.results.elapsed_time for p in programs_with_error)
  print("Time spent on errored programs", wasted_time)
  print("% of programs that errored", 
    len(programs_with_error) / len(programs),
    len(programs_with_error), len(programs))

  if input("Delete data for programs with error? ['yes']") == "yes":
    evaluation_folder = f"pickles/{experiment_id}_evaluations/"
    deleted_evaluation_folder = f"pickles/{experiment_id}_evaluations_DELETED/"
    if not os.path.exists(deleted_evaluation_folder):
        os.mkdir(deleted_evaluation_folder)
    for p in programs_with_error:
      # TODO: Fix the hardcoded ID below
      file_name = f"{p.curiosity_program.program_id}_{231}.pickle.gz"
      print("rename", p)
      os.rename(
        f"{evaluation_folder}{file_name}",
        f"{deleted_evaluation_folder}{file_name}"
      )

def smallest_programs_above_threshold(data, threshold):
  valid_data = [p for p in data if p.stats["mean_performance"] > threshold]
  smallest_size = min(len(p.program.forward_program) for p in valid_data)
  smallest_programs = [p for p in valid_data if len(p.program.forward_program) == smallest_size]
  print(f"There are {len(smallest_programs)} programs of length {smallest_size} that got a score above {threshold}")
  return smallest_programs

def print_program_data(program_data: ProgramData, experiment_id):
  print("# -----------------")
  print(f"# Experiment: {experiment_id}")
  print(f"# Index {program_data.index}")
  print(f"# {program_data.stats}")
  print("# ------")
  print(program_data.curiosity_program)
  if program_data.reward_combiner_program is not None:
    print("# ------")
    print(program.reward_combiner_program)
  print("# -----------------")

def _total_steps(trials_rollouts_episode_lengths):
  total = 0
  for t in trials_rollouts_episode_lengths:
    for r in t:
      total += sum(r)
  return total

def _view_programs(programs):
  programs = [p for p in programs if "AddToLoss" in str(p)]
  print("matching programs", len(programs))
  for i, p in enumerate(random.sample(programs, 1)):
    p.visualize_program_as_graph(i)

def place_programs_in_buckets(data, bins=50, data_per_min=4):
  data = [d for d in data if d.stats]

  scores = [d.stats["mean_performance"] for d in data]
  mn, mx = min(scores), max(scores)
  width = (mx - mn) / bins

  p_indices = []
  random.seed(1)

  for b in range(bins):
    low = mn + b * width
    high = low + width
    data_in_bucket = [d for d in data if d.stats["mean_performance"]
                      >= low and d.stats["mean_performance"] < high]
    print(low, high)
    selected_data = random.sample(data_in_bucket, min(data_per_min, len(data_in_bucket)))
    p_indices += [d.index for d in selected_data]

  print("Bucketed programs", len(p_indices))
  random.shuffle(p_indices)
  return p_indices

if __name__ == "__main__":
    main()
