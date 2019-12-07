"""
Takes a set of programs and their pre-computed result data from running those 
programs on an agent in some environment, simulate how efficient the program selection 
process would have been if we had used early termination & intelligent program selection.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

from mlca.scripts.analyze_synthesized_programs import load_curiosity_program_data, ProgramData, _stats_for_program, load_reward_combiner_program_data, _total_steps
from mlca.test_synthesized_programs_experiments import TspExperimentList, TspParams
from mlca.predict_performance_experiments import PredictPerformanceExperimentList, PredictPerformanceParams
from mlca.search_programs import EarlyTerminationBatchData, search_with_score_prediction, rollout_timestep_pruning_hook_fn, _select_next_program_data_update_with_program_result_diversity, _select_next_program_batch_diversity, _select_next_program_batch_random, _select_next_program_batch_regressor, _select_next_program_preprocess_data_diversity, _none
from mlca.run_agent import Timestep, Trial, Rollout, Reward, TrialList, RolloutList, EpisodeList, ProgramTestResultData
from mlca.diversity.density_peaks import evaluate_diversity_selection_choices
import mlca.helpers.util
import mlca.helpers.config
from mlca.search_program_experiments import SearchExperimentList, SearchParams

INTERMEDIATE_SCORE_TOP_N_FRACTION = .1

def main():
  parser = mlca.helpers.config.argparser()
  args = parser.parse_args()
  experiment_id = args.experiment_id

  search_params = SearchExperimentList[experiment_id]
  tsp_params = TspExperimentList[search_params.TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID]
  predict_performance_params = PredictPerformanceExperimentList[search_params.PREDICT_PERFORMANCE_EXPERIMENT_ID]

  with search_params:
    with tsp_params:
      with predict_performance_params:
        # =====================
        # Setup programs  
        # =====================

        if TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.CURIOSITY_SEARCH:
          data, _, _, _, _, _, _, _, _, _, _,  = load_curiosity_program_data(
              TspParams.current().CURIOSITY_PROGRAMS_NAME,
              TspParams.current().REWARD_COMBINER_PROGRAMS_NAME,
              SearchParams.current().TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID,
              TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID)
        elif TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.REWARD_COMBINER_SEARCH:
          data, _, _, _, _, _, _, _, _, _, _,  = load_reward_combiner_program_data(
              TspParams.current().CURIOSITY_PROGRAMS_NAME,
              TspParams.current().REWARD_COMBINER_PROGRAMS_NAME,
              SearchParams.current().TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID,
              TspParams.current().FIXED_CURIOSITY_PROGRAM_ID)

        data = [d for d in data if d.stats]
        programs = [d.curiosity_program for d in data]
        program_to_data = {p.program_id: d for p, d in zip(programs, data)}

        select_next_program_batch_fn = {
          "RANDOM": _select_next_program_batch_random,
          "SORT": _select_next_program_batch_regressor,
          "DIVERSITY": _select_next_program_batch_diversity
        }[SearchParams.current().BATCH_SELECTION]

        select_next_program_preprocess_data_fn = {
          "RANDOM": _none,
          "SORT": _none,
          "DIVERSITY": _select_next_program_preprocess_data_diversity
        }[SearchParams.current().BATCH_SELECTION]

        select_next_program_data_update_with_program_result_fn = {
          "RANDOM": _none,
          "SORT": _none,
          "DIVERSITY": _select_next_program_data_update_with_program_result_diversity
        }[SearchParams.current().BATCH_SELECTION]

        # =====================
        # Run NUM_SEARCHES searches
        # =====================

        most_possible_steps = TspParams.current().NUM_ROLLOUTS_PER_TRIAL * \
            TspParams.current().STEPS_PER_ROLLOUT * \
            TspParams.current().NUM_TRIALS_PER_PROGRAM * \
            len(program_to_data)

        print(f"Simulating search using {len(program_to_data)} pre-cached results.")
        search_scores = []
        all_intermediate_scores = []
        all_intermediate_num_programs_per_machine_evaluted = []
        for search_i in range(SearchParams.current().NUM_SEARCHES):

          print("----------------------------")
          print(f"Start search {search_i}/{SearchParams.current().NUM_SEARCHES}")
          print("----------------------------")

          mlca.helpers.util.set_random_seed(search_i)
          random_programs = list(programs)
          random.shuffle(random_programs)

          def get_pre_evaluated_programs_fn():
            return [], []

          intermediate_scores = []
          intermediate_num_programs_per_machine_evaluted = []

          def post_batch_hook_fn(evaluated_programs_data):
            intermediate_scores.append(_num_evaluated_programs_in_top_n_percent(
                evaluated_programs_data, program_to_data,  INTERMEDIATE_SCORE_TOP_N_FRACTION)  / \
                  math.floor(INTERMEDIATE_SCORE_TOP_N_FRACTION * len(program_to_data)))
            intermediate_num_programs_per_machine_evaluted.append(len(evaluated_programs_data))

          target_num_jobs_running = 64

          evaluate_program_fn_extra_args = (program_to_data, search_params)
          evaluated_programs_data = search_with_score_prediction(
            random_programs, 
            get_pre_evaluated_programs_fn, 
            target_num_jobs_running,
            simulate_evaluate_program_fn,
            rollout_timestep_pruning_hook_fn,
            select_next_program_batch_fn, 
            select_next_program_preprocess_data_fn, 
            select_next_program_data_update_with_program_result_fn,
            post_batch_hook_fn,
            get_early_termination_batch_data_fn, 
            search_params, 
            tsp_params,
            evaluate_program_fn_extra_args, use_threads=True)

          evaluated_programs_order = [d.curiosity_program for d in evaluated_programs_data]

          score = _average_rank_of_top_n_percent_of_programs(
              evaluated_programs_order, program_to_data)
          search_scores.append(score)

          all_intermediate_scores.append(intermediate_scores)
          all_intermediate_num_programs_per_machine_evaluted.append(
              intermediate_num_programs_per_machine_evaluted)

          print(
            f"""
            _average_rank_of_top_n_percent_of_programs {score} 
            avg score {np.array(search_scores).mean()}
            Num programs ran {len(evaluated_programs_data)}
            Num programs pruned {len([d for d in evaluated_programs_data if d.results.early_terminated])}
            Num of steps run {_total_num_steps_run(evaluated_programs_data)}
            % of top {INTERMEDIATE_SCORE_TOP_N_FRACTION} programs found {intermediate_scores[-1]}
            time saved {1 - (_total_num_steps_run(evaluated_programs_data) / most_possible_steps)}
            time saved w. missing programs {(1 - (_total_num_steps_run(evaluated_programs_data) / (most_possible_steps  * intermediate_scores[-1])))}
            """)

          evaluated_program_ids = [
            d.curiosity_program.program_id
            for d in evaluated_programs_data
          ]
          evaluated_programs = [d.curiosity_program for d in evaluated_programs_data]
          print(evaluated_program_ids)
          print("num_removed_programs", evaluate_diversity_selection_choices(
              evaluated_programs,
              data
            ))

        if SearchParams.current().ENABLE_EARLY_TERMINATION:
          early_termination_batch_data = get_early_termination_batch_data_fn(
              evaluated_programs_data
          )
        else:
          early_termination_batch_data = None

        _plot_program_evaluations(
          evaluated_programs_data, early_termination_batch_data)

        # _plot_cutoffs(early_termination_batch_data)

        # _plot_pruning_distribution(evaluated_programs_data)

        _plot_all_intermediate_scores(
            all_intermediate_num_programs_per_machine_evaluted, all_intermediate_scores,
            f"% of Top {INTERMEDIATE_SCORE_TOP_N_FRACTION} Programs Found vs # Evaluated",
            f"% of top {INTERMEDIATE_SCORE_TOP_N_FRACTION} programs found")

"""
Simulate the evaluation of a program in an environment by stepping through the execution
trace and checking if early-termination would terminate the program at each timestep.
"""
def simulate_evaluate_program_fn(
    program, rollout_timestep_pruning_hook_fn,
    early_termination_batch_data, selected_index, tsp_params, extra_curiosity_programs,
    evaluate_program_fn_extra_args, result_pipe_connection):

  program_to_data, params = evaluate_program_fn_extra_args

  # Simulate scanning through each trial sequentially, and through each timestep in the trial.
  trials_rollouts_mean_reward: TrialList[RolloutList[Reward]] = []
  trials_rollouts_episode_lengths: TrialList[RolloutList[EpisodeList[int]]] = []
  trials_rollouts_episode_end_rewards: TrialList[RolloutList[EpisodeList[Reward]]] = []
  data = program_to_data[program.program_id]

  def create_results_program_data(early_terminated):
    trials_rollouts_mean_reward: TrialList[EpisodeList[Reward]] = []
    for t in trials_rollouts_episode_end_rewards:
      trial_final_rewards: EpisodeList[Reward] = []
      trials_rollouts_mean_reward.append(trial_final_rewards)
      for r in t:
        trial_final_rewards += r
    results = ProgramTestResultData(
      trials_rollouts_mean_reward,
      trials_rollouts_episode_lengths,
      trials_rollouts_episode_end_rewards,
      False,
      None,
      early_terminated,
      early_termination_batch_data is not None,
      selected_index,
      time.time(),
      time.time(),
      0,
      None,
      None,
      None,
      None,
    )
    return ProgramData(
        program.program_id, program, None, results,
        _stats_for_program(results)
    )

  for trial, (trial_episode_final_rewards, trial_rollouts_episode_lengths) in enumerate(zip(
          data.results.trials_rollouts_episode_end_rewards,
          data.results.trials_rollouts_episode_lengths)):

    if early_termination_batch_data is not None:
      for timestep_cap in range(
              SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY,
              TspParams.current().STEPS_PER_ROLLOUT,
              SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY):
        # Simulate the average episode ending rewards we would have seen at this timestep.
        rollouts_episode_end_rewards: List[List[Reward]] = []
        rollouts_episode_lengths: List[List[int]] = []

        assert len(trial_episode_final_rewards) == len(trial_rollouts_episode_lengths), (
            len(trial_episode_final_rewards), len(trial_rollouts_episode_lengths))

        for rollout, (rollout_final_rewards, rollout_episode_lengths) in enumerate(
                zip(trial_episode_final_rewards, trial_rollouts_episode_lengths)):
          timestep = 0
          assert len(rollout_final_rewards) == len(rollout_episode_lengths)
          episode_end_rewards: List[Reward] = []
          episode_lengths: List[int] = []
          rollouts_episode_end_rewards.append(episode_end_rewards)
          rollouts_episode_lengths.append(episode_lengths)

          for final_reward, episode_length in zip(rollout_final_rewards, rollout_episode_lengths):
            timestep += episode_length
            if timestep > timestep_cap:
              break
            else:
              episode_end_rewards.append(final_reward)
              episode_lengths.append(episode_length)

        if len(rollouts_episode_end_rewards) > 0:
          if "MiniGrid" in TspParams.current().ENVIRONMENT:
            mean_episode_end_reward = np.array(
                [np.array(r).max() if len(r) > 0 else math.nan
                  for r in rollouts_episode_end_rewards]
            ).max()
          else:
            mean_episode_end_reward = np.array(
                [np.array(r).mean() for r in rollouts_episode_end_rewards]
            ).mean()
        else:
          mean_episode_end_reward = math.nan

        prune = len(episode_end_rewards) > 0 and rollout_timestep_pruning_hook_fn(
            trial, timestep_cap, mean_episode_end_reward, early_termination_batch_data)

        if prune:
          trials_rollouts_episode_lengths.append(rollouts_episode_lengths)
          trials_rollouts_episode_end_rewards.append(
              rollouts_episode_end_rewards)

          result_pipe_connection.send(
              create_results_program_data(early_terminated=True))
          return

    # Didn't prune, just add everything:
    trials_rollouts_episode_end_rewards.append(
        data.results.trials_rollouts_episode_end_rewards[trial])
    trials_rollouts_episode_lengths.append(
        data.results.trials_rollouts_episode_lengths[trial]
    )
  
  result_pipe_connection.send(create_results_program_data(early_terminated=False))
  return

"""
Compute the early-termination cutoff at every (trial, timetstep) by looking
at how well the top n programs performed at at that point.
"""
def get_early_termination_batch_data_fn(evaluated_programs_data) -> EarlyTerminationBatchData:
  evaluated_programs_data = [d for d in evaluated_programs_data if d is not None and d.stats is not None]
  
  if len(evaluated_programs_data) == 0:
    return None
  else:
    program_data_by_perf = sorted(evaluated_programs_data,
                                  key=lambda d: d.stats["mean_performance"])
    best_program_data = program_data_by_perf[-SearchParams.current().NUM_BEST_PROGRAMS:]

    timestep_program_mean_performance: Dict[Timestep, List[Reward]] = {}
    timestep_program_stds_across_trials: Dict[Tuple[Trial, Timestep], List[float]] = {}

    # Find the average end-of-episode reward at each (trial, timestep_cap), averaged over all trials and timesteps for the top 10 agents
    for data in tqdm(best_program_data, "get_early_termination_batch_data_fn"):
      program_timestep_rewards: Dict[Timestep, List[Reward]] = {}
      for trial, (trial_episode_final_rewards, trial_rollouts_episode_lengths) in enumerate(zip(
              data.results.trials_rollouts_episode_end_rewards, data.results.trials_rollouts_episode_lengths)):
        for timestep_cap in range(
                SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY, 
                TspParams.current().STEPS_PER_ROLLOUT, 
                SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY):
          # Find average reward of finished episodes by this timestep
          trial_avg_episode_end_rewards = []
          for rollout, (rollout_final_rewards, rollout_episode_lengths) in enumerate(
                  zip(trial_episode_final_rewards, trial_rollouts_episode_lengths)):
              rollout_episode_end_rewards = []

              timestep = 0
              for final_reward, episode_length in zip(rollout_final_rewards, rollout_episode_lengths):
                timestep += episode_length
                if timestep > timestep_cap:
                  break
                else:
                  rollout_episode_end_rewards.append(final_reward)

              if len(rollout_episode_end_rewards) > 0:
                if "MiniGrid" in TspParams.current().ENVIRONMENT:
                  trial_avg_episode_end_rewards.append(np.array(rollout_episode_end_rewards).max())
                else:
                  trial_avg_episode_end_rewards.append(np.array(rollout_episode_end_rewards).mean())

          if len(trial_avg_episode_end_rewards) >  0:
            if "MiniGrid" in TspParams.current().ENVIRONMENT:
              avg_episode_end_rewards = np.array(trial_avg_episode_end_rewards).max()
            else:
              avg_episode_end_rewards = np.array(trial_avg_episode_end_rewards).mean()
          else:
            avg_episode_end_rewards = np.nan

          if avg_episode_end_rewards != np.nan:
            if timestep_cap not in program_timestep_rewards:
              program_timestep_rewards[timestep_cap] = []
            program_timestep_rewards[timestep_cap].append(avg_episode_end_rewards)

      for timestep_cap in program_timestep_rewards:
        # Track stdevs within programs (across trials) on each timestep
        for trial in range(TspParams.current().NUM_TRIALS_PER_PROGRAM):
          if (trial, timestep_cap) not in timestep_program_stds_across_trials:
            timestep_program_stds_across_trials[(trial, timestep_cap)] = []
            
          timestep_program_stds_across_trials[(trial, timestep_cap)].append(
            np.array(program_timestep_rewards[timestep_cap]).std() / math.sqrt(
              min(trial + 1, len(program_timestep_rewards[timestep_cap]))
            )
          )

        # Track mean performance
        if timestep_cap not in timestep_program_mean_performance:
            timestep_program_mean_performance[timestep_cap] = []
        timestep_program_mean_performance[timestep_cap].append(
            np.array(program_timestep_rewards[timestep_cap]).mean()
        )

    # Set the caps for each (trial, timestep) to the mean performance seen by the best programs - 2 * variance
    trial_timestep_caps = {
        (trial, timestep_cap): \
          # Average program performance on this timestep
          np.array(timestep_program_mean_performance[timestep_cap]).mean() \
          # Minus differences between programs
          - SearchParams.current().NUM_STDEVS_DOWN * np.array(timestep_program_mean_performance[timestep_cap]).std()
          # Minus the average difference of program's trials, normalizing using the # of timesteps.
          - np.array(timestep_program_stds_across_trials[(trial, timestep_cap)]).mean() 
        for trial in range(TspParams.current().NUM_TRIALS_PER_PROGRAM)
        for timestep_cap in timestep_program_mean_performance.keys()
    }

    return trial_timestep_caps

def _plot_cutoffs(early_termination_batch_data):
  cutoffs = []
  for key in early_termination_batch_data:
    cutoffs.append(early_termination_batch_data[key])
  plt.plot(cutoffs)
  plt.show()

def _plot_pruning_distribution(evaluated_programs_data):
  steps = [_total_steps(d.results.trials_rollouts_episode_lengths)
           for d in evaluated_programs_data]
  plt.hist(steps, 50)
  plt.xlabel("# steps evaluated")
  plt.ylabel("# programs")
  plt.show()

"""
Helper that plots how program evaluations evolve over time.
"""
def _plot_program_evaluations(evaluated_programs_data, early_termination_batch_data):
  end_times = []
  end_rewards = []

  max_evaluation_index = max(d.results.selected_index for d in evaluated_programs_data)

  random.shuffle(evaluated_programs_data)
  for d in tqdm(evaluated_programs_data, "Plotting program evaluations"):
    evaluation_index = d.results.selected_index
    # Loop through episode, keeping track of lengths and rewards
    reward = 0
    episode_rewards = []
    time = 0
    # TODO: Dedup this code
    for trial, (trial_rollouts_episode_lengths, trial_episode_final_rewards) in enumerate(zip(
        d.results.trials_rollouts_episode_lengths,
        d.results.trials_rollouts_episode_end_rewards)):
        rewards = []
        times = []
        for timestep_cap in range(
                SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY,
                TspParams.current().STEPS_PER_ROLLOUT,
                SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY):
          # Simulate the average episode ending rewards we would have seen at this timestep.
          rollouts_episode_end_rewards = []
          rollouts_episode_lengths = []
          reached_timestep = False

          for rollout, (rollout_final_rewards, rollout_episode_lengths) in enumerate(
                  zip(trial_episode_final_rewards, trial_rollouts_episode_lengths)):
            timestep = 0
            assert len(rollout_final_rewards) == len(rollout_episode_lengths)
            episode_end_rewards = []
            episode_lengths = []
            rollouts_episode_end_rewards.append(episode_end_rewards)
            rollouts_episode_lengths.append(episode_lengths)

            for final_reward, episode_length in zip(rollout_final_rewards, rollout_episode_lengths):
              timestep += episode_length
              if timestep > timestep_cap:
                reached_timestep = True
                break
              else:
                episode_end_rewards.append(final_reward)
                episode_lengths.append(episode_length)

          if reached_timestep:
            if "MiniGrid" in TspParams.current().ENVIRONMENT:
              mean_episode_end_reward = np.array(
                  [np.array(r).max() for r in rollouts_episode_end_rewards]
              ).max()
            else:
              mean_episode_end_reward = np.array(
                  [np.array(r).mean() for r in rollouts_episode_end_rewards]
              ).mean()
          else:
            break

          times.append(trial * TspParams.current().STEPS_PER_ROLLOUT + timestep_cap)
          rewards.append(mean_episode_end_reward)

        if d.results.get("had_early_termination_data", True): # evaluation_index > SearchParams.current().PROGRAMS_PER_BATCH):
          color = evaluation_index / max_evaluation_index

          plt.plot(times, rewards, alpha=.01, color=(color, 1 - color, 0))

    # Note this intentionally happens outisde of the for loop, after the last trial
    if len(times) > 0:
      end_times.append(times[-1])
      end_rewards.append(rewards[-1])
    # else: 
    #   plt.plot(times, rewards, alpha=.05, color="gray")

  if early_termination_batch_data: 
    cutoff_times = []
    cutoff_levels = []
    for key in early_termination_batch_data:
      trial, timestep = key
      cutoff_times.append(trial * TspParams.current().STEPS_PER_ROLLOUT + timestep)
      cutoff_levels.append(early_termination_batch_data[key])
    plt.plot(cutoff_times, cutoff_levels, color='yellow', alpha=.5)

  plt.scatter(end_times, end_rewards, color='blue', s=1, alpha=.5, zorder=10)

  plt.xlabel("# steps evaluated")
  plt.ylabel("avg episode reward")
  plt.show()


def _plot_all_intermediate_scores(all_num_programs_per_machine_evaluted, all_intermediate_scores, title, ylabel):
  # print(all_num_programs_per_machine_evaluted)
  mean_intermediate_score = np.array(all_intermediate_scores).mean(axis=0)
  plt.plot(all_num_programs_per_machine_evaluted[0], mean_intermediate_score)
  plt.xlabel('Num programs evaluated')
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()

def _total_num_steps_run(evaluated_programs_data):
  steps = 0
  for p in evaluated_programs_data:
    for t in p.results.trials_rollouts_episode_lengths:
      for r in t:
        for episode_len in r:
          steps += episode_len
  return steps


def _average_rank_of_top_n_percent_of_programs(evaluated_programs_order, program_to_data, top_n_fraction=.1):
  # Compute the average rank of the top n% of programs
  # (Rank = position in which the program was selected in the search)
  programs_with_rank = list(enumerate(evaluated_programs_order))
  sorted_programs = sorted(programs_with_rank, 
    key = lambda rank_and_p: program_to_data[rank_and_p[1].program_id].stats["mean_performance"]
  )
  top_n_programs = sorted_programs[-int(top_n_fraction * len(sorted_programs)):]
  rank_of_top_n_programs = np.array([rank_and_p[0] for rank_and_p in top_n_programs])
  return rank_of_top_n_programs.mean()

def _num_evaluated_programs_in_top_n_percent(evaluated_programs_data, original_program_to_data, top_n_fraction=.1):
  # Compute the average rank of the top n% of programs
  # (Rank = position in which the program was selected in the search)
  
  sorted_programs = sorted([program_id for program_id in original_program_to_data.keys()],
                           key=lambda program_id: original_program_to_data[program_id].stats["mean_performance"]
                           )
  top_n_programs_ids = sorted_programs[-int(top_n_fraction * len(sorted_programs)):]

  return len(
    [1 for d in evaluated_programs_data \
     if d.curiosity_program.program_id in top_n_programs_ids \
        and not d.results.early_terminated])

if __name__ == "__main__":
    main()
