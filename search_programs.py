"""
Run through a list of synthesized programs and evaluate them in an 
environment. Potentially uses early program stopping, intelligent program
re-ordering, etc, if those settings are enabled in the current experiment.

See test_synthesized_programs.py for the code that triggers this search.

"""

import math
import numpy as np
import random
import multiprocessing
import threading
import time
import sklearn.neighbors
from tqdm import tqdm
import itertools
from typing import List, Callable, Tuple
from typing import List, Dict, Tuple, Optional

import mlca.helpers.config
import mlca.helpers.debug
import mlca.helpers.util

from mlca.search_program_experiments import SearchParams
from mlca.test_synthesized_programs_experiments import TspParams
from mlca.program import Program
from mlca.predict_performance import ProgramFeatureVector, PredictPerformanceParams, PredictPerformanceExperimentList, get_predict_performance_regressor, prob_programs_above_perf_threshold_knn, prob_programs_above_perf_threshold_gp, program_as_feature_vector_diversity, program_as_feature_vector_predict_performance
from mlca.run_agent import Timestep, Trial, Rollout, Reward, TrialList, RolloutList, EpisodeList, ProgramTestResultData
import mlca.operations as operations

EarlyTerminationBatchData = Optional[Dict[
  Tuple[Trial, Timestep], # (trial, timestep_cap)
  Reward  # cutoff
]]

"""
Given a list of programs, evaluate every program in an environment.

Arguments:
    programs -- The list of programs to search
    get_pre_evaluated_programs_fn -- The function that loads any evaluation data from a previous run of this call.
    target_num_jobs_running -- The number of jobs (processes) that should simultaneously be running to maximize your CPU/GPU.
    evaluate_program_fn -- The function that actually evaluates a program.
    rollout_timestep_pruning_hook_fn -- The function passed into evaluate_program_fn that checks if a program evaluation should terminate.
    select_next_program_batch_fn -- The function that selects the next batch of programs to evaluate.
    post_batch_hook_fn -- The function that calls whenever a program batch is finished being evaluated.
    get_early_termination_batch_data_fn -- The function that takes the currently evaluated programs and computes early termination thresholds.
    search_params, tsp_params -- The experiment parameters.
    evaluate_program_fn_extra_args -- Extra arguments to pass to evaluate_program_fn
"""
def search_with_score_prediction(
        programs: List[Program], 
        get_pre_evaluated_programs_fn: Callable[[], Tuple[List[Program], List[ProgramTestResultData]]], 
        target_num_jobs_running: int,
        evaluate_program_fn, 
        rollout_timestep_pruning_hook_fn: Callable[[int, int, int, EarlyTerminationBatchData], bool],
        select_next_program_batch_fn,
        select_next_program_preprocess_data_fn,
        select_next_program_data_update_with_program_result_fn,
        post_batch_hook_fn,
        get_early_termination_batch_data_fn: Callable[[List[ProgramTestResultData]], EarlyTerminationBatchData], 
        search_params: SearchParams, 
        tsp_params: TspParams, 
        evaluate_program_fn_extra_args,
        use_threads=False):
  evaluated_programs_data = []
  unevaluated_programs = list(programs)
  programs_by_id = {p.program_id: p for p in programs}

  early_termination_batch_data = None

  # If previously started this search, reload those results
  already_evaluated_programs, batch_results = get_pre_evaluated_programs_fn()
  already_evaluated_programs = already_evaluated_programs[:1000]
  batch_results = batch_results[:1000]
  received_program_batch_with_no_programs = False
  evaluated_programs_data.extend(batch_results)
  for program in already_evaluated_programs:
      unevaluated_programs.remove(program)

  select_next_program_data = select_next_program_preprocess_data_fn(
    programs)
      
  currently_running_programs: List[Program] = []
  select_next_program_job = None
  select_next_program_pipe = None

  next_programs_to_run: List[Program] = []

  MIN_CURRENTLY_RUNNING_PROGRAMS_BUFFER = SearchParams.current().PROGRAMS_PER_BATCH

  print(f"WILL RUN {target_num_jobs_running} JOBS IN PARALLEL")

  ctx = multiprocessing.get_context('spawn')
  Job = threading.Thread if use_threads else ctx.Process

  # Search through the rest of the unevaluated programs
  while (len(unevaluated_programs) > 0 and not received_program_batch_with_no_programs) \
      or len(currently_running_programs) > 0 or len(next_programs_to_run) > 0:
    # print("unevaluated_programs", len(unevaluated_programs))

    # Make sure we have target_num_jobs_running programs running
    while len(currently_running_programs) < target_num_jobs_running and len(next_programs_to_run) > 0:
      program = next_programs_to_run.pop()
      selected_index = len(evaluated_programs_data) + len(currently_running_programs)

      pipe = ctx.Pipe(False)
      _, result_pipe_connection = pipe

      if TspParams.current().COMPUTE_PROGRAM_CORRELATIONS:
        extra_curiosity_programs = random.sample(
          programs,
          TspParams.current().COMPUTE_PROGRAM_CORRELATIONS_PROGRAMS_PER_BATCH)
      else: 
        extra_curiosity_programs = None
      
      process = Job(target=evaluate_program_fn, args=(
        program, rollout_timestep_pruning_hook_fn, 
        early_termination_batch_data, selected_index, tsp_params,
        extra_curiosity_programs, evaluate_program_fn_extra_args,
        result_pipe_connection))
      process.start()

      currently_running_programs.append((process, pipe))

      print("Started up new job. Currently running programs", len(currently_running_programs), 
            "Active children", threading.active_count() if use_threads else len(multiprocessing.active_children()))

    # Collect data from finished jobs
    for job in list(currently_running_programs):
      process, pipe = job
      if not process.is_alive():
        currently_running_programs.remove(job)
        result = pipe[0].recv()
        evaluated_programs_data.append(result)
        select_next_program_data_update_with_program_result_fn(
          programs, select_next_program_data, result
        )
        # evaluated_programs.append(program)
        pipe[0].close()
        pipe[1].close()
        if not use_threads:
          process.close()
        print(f"Finished job. Num results {len(evaluated_programs_data)}. Remaining# running jobs: {len(currently_running_programs)}")

    # Check if we need to select the next program batch & update early termination data
    if select_next_program_job is None:
      if len(currently_running_programs) + len(next_programs_to_run) < MIN_CURRENTLY_RUNNING_PROGRAMS_BUFFER and len(unevaluated_programs) > 0:
        print("Start looking for new jobs")

        pipe = ctx.Pipe(False)
        receiver, result_pipe_connection = pipe

        process = Job(target=select_next_programs_and_early_termination_data, args=(
          select_next_program_batch_fn, get_early_termination_batch_data_fn, evaluated_programs_data, select_next_program_data,
          unevaluated_programs, search_params, tsp_params, result_pipe_connection
        ), daemon=True)
        process.start()

        select_next_program_job = process
        select_next_program_pipe = pipe

        post_batch_hook_fn(evaluated_programs_data)
    elif select_next_program_pipe[0].poll(): # select_next_program_job.is_alive():
      print("Selected new jobs")
      program_batch, early_termination_batch_data = select_next_program_pipe[0].recv()
      if len(program_batch) == 0:
        received_program_batch_with_no_programs = True
      program_batch = [programs_by_id[p.program_id] for p in program_batch]
      next_programs_to_run.extend(program_batch)
      for program in program_batch:
        unevaluated_programs.remove(program)
      print("unevaluated_programs", len(unevaluated_programs))

      select_next_program_pipe[0].close()
      # Don't close select_next_program_pipe[1], the sender closed it.
      select_next_program_job.join()
      if not use_threads:
        select_next_program_job.close()
      
      select_next_program_job = None

    time.sleep(1)

  print("All done!", len(evaluated_programs_data))
  return  evaluated_programs_data

def select_next_programs_and_early_termination_data(
    select_next_program_batch_fn, get_early_termination_batch_data_fn, evaluated_programs_data, select_next_program_data,
    unevaluated_programs, search_params, tsp_params, result_pipe_connection):
  predict_performance_params = PredictPerformanceExperimentList[search_params.PREDICT_PERFORMANCE_EXPERIMENT_ID]
  with search_params:
    with tsp_params:
      with predict_performance_params:
        program_batch = select_next_program_batch_fn(
            evaluated_programs_data,
            unevaluated_programs,
            select_next_program_data
          )

        if SearchParams.current().ENABLE_EARLY_TERMINATION:
          early_termination_batch_data = get_early_termination_batch_data_fn(
              evaluated_programs_data
          )
        else:
          early_termination_batch_data = None

        print("Piping data back")
        result_pipe_connection.send((program_batch, early_termination_batch_data))
        result_pipe_connection.close()
        print("Returned")

def rollout_timestep_pruning_hook_fn(
        trial, timestep, avg_episode_end_reward, early_termination_batch_data):
  if not early_termination_batch_data:
    return False
  elif (trial, timestep) not in early_termination_batch_data:
    print("Missing", (trial, timestep))
    return False
  elif math.isnan(early_termination_batch_data[(
      trial, timestep)]) or math.isnan(avg_episode_end_reward):
    return False
  else:
    prune = avg_episode_end_reward < early_termination_batch_data[(
        trial, timestep)]
    return prune

def _select_next_program_batch_random(
    evaluated_programs_data, unevaluated_programs, select_next_program_data):
  random_programs = list(unevaluated_programs)
  random.shuffle(random_programs)
  return random_programs[:SearchParams.current().PROGRAMS_PER_BATCH]

def _select_next_program_batch_regressor(
    evaluated_programs_data, unevaluated_programs, select_next_program_data):

  evaluated_programs_data = [d for d in evaluated_programs_data if d is not None and d.results is not None and d.stats is not None and not math.isnan(d.stats["mean_performance"])]

  if len(evaluated_programs_data) <= 10:
    return _select_next_program_batch_random(evaluated_programs_data, unevaluated_programs, select_next_program_data)
  else:
      regr = get_predict_performance_regressor()

      train_X = np.array([program_as_feature_vector_predict_performance(d.curiosity_program)
        for d in evaluated_programs_data])

      train_y = np.array([d.stats["mean_performance"]
                          for d in evaluated_programs_data])

      regr.fit(train_X, train_y)
      test_X = np.array([program_as_feature_vector_predict_performance(program)
                          for program in unevaluated_programs])
      scores = regr.predict(test_X)
      
      if SearchParams.current().NOVELTY_BONUS:
        # TODO: Include this into the regressor, so not tied to kneighbors
        kneighbors_dist, kneighbors_ind = regr.kneighbors(
          test_X, n_neighbors=PredictPerformanceParams.current().NEIGHBORS)

        kneighbors = train_X[kneighbors_ind] # (num_test, num_neighbors, num_features)
        test_X_e = np.expand_dims(test_X, 1) # (num_test, 1, num_features)

        diff_with_kneighbors = kneighbors.astype(
            np.float) - test_X_e.astype(np.float)  # (num_test, num_neighbors, num_features)
        if SearchParams.current().NOVELTY_DISTANCE == "L2":
            dist_to_kneighbors = np.linalg.norm(
                diff_with_kneighbors, ord=None, axis=2)  # (num_test, num_neighbors)
        elif SearchParams.current().NOVELTY_DISTANCE == "L1":
            dist_to_kneighbors = np.linalg.norm(
                diff_with_kneighbors, ord=None, axis=1)  # (num_test, num_neighbors)
        elif SearchParams.current().NOVELTY_DISTANCE == "L1Normalized":
            dist_to_kneighbors = np.linalg.norm(
                diff_with_kneighbors, ord=None, axis=1) / np.sum(test_X_e, axis=2)

        avg_dist_to_kneighbors = dist_to_kneighbors.mean(axis=1)  # (num_test_
        novelty_scores = avg_dist_to_kneighbors

        total_scores = scores + SearchParams.current().NOVELTY_WEIGHT * novelty_scores

      else:
        total_scores = scores

      num_promising = math.floor( ( 1 - SearchParams.current().EPSILON_RANDOM_PROGRAM_SELECTION ) * SearchParams.current().PROGRAMS_PER_BATCH)
      num_random = math.ceil( SearchParams.current().EPSILON_RANDOM_PROGRAM_SELECTION * SearchParams.current().PROGRAMS_PER_BATCH)

      random_programs_to_try = set(random.sample(list(range(len(unevaluated_programs))), min(len(unevaluated_programs), num_random)))

      indicies = sorted(range(len(unevaluated_programs)),
                      key=lambda i: total_scores[i] + (math.inf if i in random_programs_to_try else 0))

      return list(unevaluated_programs[i] for i in indicies[-SearchParams.current().PROGRAMS_PER_BATCH:])

def _select_next_program_batch_diversity(
  evaluated_programs_data: List[ProgramTestResultData], 
  unevaluated_programs: List[Program],
  select_next_program_data):

  if len(evaluated_programs_data) <= 10:
    return _select_next_program_batch_random(
      evaluated_programs_data, unevaluated_programs, select_next_program_data)
  else:
    print("Start _select_next_program_batch_diversity")
    start = time.time()

    # For every point
      # Compute the min performance it needs to be above the delta and perf thresholds
      # Compute the prob that the point is above that threshold

    program_id_to_feature_vector_diversity = select_next_program_data["fv_diversity"]
    program_id_to_feature_vector_performance = select_next_program_data["fv_performance"]

    # Compute program features
    # evaluated_program_features = np.array([
    #     program_id_to_feature_vector_diversity[d.curiosity_program.program_id] 
    #       for d in evaluated_programs_data
    # ]).astype(np.double)

    unevaluated_program_features = np.array([
        program_id_to_feature_vector_diversity[p.program_id]
          for p in unevaluated_programs
    ]).astype(np.double)

    # Setup prediction regressor
    performance_regressor = get_predict_performance_regressor()

    train_X = np.array([
      program_id_to_feature_vector_performance[d.curiosity_program.program_id]
      for d in evaluated_programs_data])

    unevaluated_program_features_prediction = np.array([
        program_id_to_feature_vector_performance[p.program_id] for p in unevaluated_programs
    ]).astype(np.double)

    train_y = np.array([d.stats["mean_performance"]
                        for d in evaluated_programs_data])

    performance_regressor.fit(train_X, train_y)

    # Setup diversity knn & parameters
    # evaluated_program_features_tree = sklearn.neighbors.KDTree(
    #   evaluated_program_features
    # )

    DELTA = SearchParams.current().DIVERSITY_DELTA # 2.5 # TspParams.
    PERF_THRESHOLD = SearchParams.current().DIVERSITY_PERF_THRESHOLD # 400 # TODO: Put in TspParams

    # probs_above_threshold: List[float] = []
    perf_thresholds: List[float] = []

    # ind_within_deltas = evaluated_program_features_tree.query_radius(
    #     unevaluated_program_features, DELTA)

    for p, fv in zip(
        unevaluated_programs, tqdm(unevaluated_program_features, "Compute diversity performance thresholds")):
    
      perf_threshold = max(
        PERF_THRESHOLD,
        select_next_program_data["program_thresholds"].get(p.program_id, -math.inf)
      )
      # if len(ind_within_delta) == 0:
      #   perf_threshold = PERF_THRESHOLD
      # else: 
      #   highest_performance_within_delta = max(
      #     evaluated_programs_data[int(i)].stats["mean_performance"] # TODO is this right
      #     for i in ind_within_delta
      #   )

        # perf_threshold = max(highest_performance_within_delta, PERF_THRESHOLD)
      perf_thresholds.append(perf_threshold)

      # p = prob_program_above_perf_threshold(fv, perf_threshold)
      # probs_above_threshold.append(p)

    if PredictPerformanceParams.current().MODEL == "KNN":
      probs_above_threshold = prob_programs_above_perf_threshold_knn(
        unevaluated_program_features_prediction,
        performance_regressor,
        perf_threshold
      )
    elif PredictPerformanceParams.current().MODEL == "GP":
      probs_above_threshold = prob_programs_above_perf_threshold_gp(
        unevaluated_program_features_prediction,
        performance_regressor,
        perf_threshold
      )
    else:
      raise RuntimeError(f"Do not support {PredictPerformanceParams.current().MODEL} with diversity.")

    # print(len(probs_above_threshold), len(unevaluated_programs))
    print("sum(probs_above_threshold)", sum(probs_above_threshold))

    program_ids_by_prob_meet_threshold = sorted(
      range(len(unevaluated_programs)), key=lambda i: -probs_above_threshold[i]
    )

    # print(probs_above_threshold)

    selected_batch: List[Program] = []
    selected_batch_feature_vectors: List[ProgramFeatureVector] = []

    skipped_program_ids = []

    # Select our batch by finding the programs that are most likely to 
    # beat their performance threshold + that aren't too close to a program
    # with a better probability
    # Note this loop goes from highest to lowest prob
    checked_programs = 0
    for i in program_ids_by_prob_meet_threshold:
      if probs_above_threshold[i] == 0:
        print(f"Out of the {len(probs_above_threshold)} remaining programs, only {checked_programs} have > 0 predicted change of beating the cutoffs. ")
        break
      else:
        program, fv = unevaluated_programs[i], unevaluated_program_features[i]

        # TODO: Dynamically build a KD tree instead?
        d = min(
          np.linalg.norm(fv - selected_fv) for selected_fv in selected_batch_feature_vectors
        ) if len(selected_batch_feature_vectors) > 0 else math.inf

        checked_programs += 1

        if d >= DELTA:
          selected_batch.append(program)
          selected_batch_feature_vectors.append(fv)
        else:
          skipped_program_ids.append(i)
      
        if len(selected_batch) >= SearchParams.current().PROGRAMS_PER_BATCH:
          break

    print(f"Iterated through {checked_programs} programs to find the batch of {len(selected_batch)} programs")

    while len(selected_batch) < SearchParams.current().PROGRAMS_PER_BATCH and len(skipped_program_ids) > 0:
      i = skipped_program_ids.pop(0)
      program, fv = unevaluated_programs[i], unevaluated_program_features[i]
      selected_batch.append(program)
      selected_batch_feature_vectors.append(fv)

    print(f"End _select_next_program_batch_diversity after {time.time() - start} s. Full batch has {len(selected_batch)} programs.")

    return selected_batch

def _select_next_program_preprocess_data_diversity(
  programs: List[Program]
):
  fv_diversity_list = [
    program_as_feature_vector_diversity(p) for p in tqdm(programs, "Pregenerate programs as feature vectors for diversity")
  ]
  return {
    "fv_diversity": {
      p.program_id: fv for p, fv in zip(programs, fv_diversity_list)
    }, 
    "fv_performance": {
      p.program_id: program_as_feature_vector_predict_performance(p) 
      for p in tqdm(programs, "Pregenerate programs as feature vectors for performance")
    },
    "kd_tree_by_fv_diversity": sklearn.neighbors.KDTree(
      np.array(fv_diversity_list)
    ),
    "program_thresholds": {}
  }

def _select_next_program_data_update_with_program_result_diversity(
  programs,
  select_next_program_data,
  result: ProgramTestResultData
):

  # Update select_next_program_data["program_thresholds"] by finding all 
  # programs within DELTA of the result feature_vector and updating their 
  # performance threshold cutoffs

  # TODO: Wire this to diversity only

  assert TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.CURIOSITY_SEARCH, "Reward search not yet supported"
  result_fv = select_next_program_data["fv_diversity"][result.curiosity_program.program_id]
  result_performance = result.stats["mean_performance"]

  ind_within_delta = select_next_program_data["kd_tree_by_fv_diversity"].query_radius(
      [result_fv], SearchParams.current().DIVERSITY_DELTA)[0]

  program_thresholds = select_next_program_data["program_thresholds"]
  for ind in ind_within_delta:
    program = programs[int(ind)]
    program_thresholds[program.program_id] = max(
      program_thresholds.get(program.program_id, -math.inf),
      result_performance
    )

def _none(*args):
  return None