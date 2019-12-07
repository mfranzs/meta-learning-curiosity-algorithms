"""
Runs an agent with a given intrinsic reward program & reward combiner program
in an environment.
"""

import pickle
import math
import numpy as np
import gzip
import gym
from gym.core import Env
import torch
import time
from tqdm import tqdm
import traceback
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import scipy.stats

from tensorboardX import SummaryWriter

import mlca.helpers.config
import mlca.helpers.debug
import mlca.helpers.util

from mlca.test_synthesized_programs_experiments import TspParams
from mlca.search_program_experiments import SearchParams, SearchExperimentList
from mlca.scripts.analyze_synthesized_programs import ProgramData, _stats_for_program
from mlca.executor import ProgramExecutionError
from mlca.internal_rewards import InternalRewardModule
from mlca.program import ProgramId

import a2c_ppo_acktr.utils 
import a2c_ppo_acktr.algo 
import a2c_ppo_acktr.arguments 
import a2c_ppo_acktr.envs 
import a2c_ppo_acktr.model 
import a2c_ppo_acktr.storage 

Timestep = int
Trial = int
Rollout = int
Reward = float
TrialList = List
RolloutList = List
EpisodeList = List

@dataclass
class ProgramTestResultData:
    trials_rollouts_mean_reward: TrialList[RolloutList[Reward]]
    trials_rollouts_episode_lengths: TrialList[RolloutList[EpisodeList[int]]]
    trials_rollouts_episode_end_rewards: TrialList[RolloutList[EpisodeList[Reward]]]

    execution_had_error: bool
    error: Optional[str]
    early_terminated: bool
    had_early_termination_data: bool

    selected_index: int # Index at which the program was selected in the search
    time: float
    start_time: float
    elapsed_time: float
    avg_stats: mlca.helpers.debug.InMemoryLoggerAverageStats

    curiosity_program_id: int
    reward_combiner_program_id: int
    device: str

    trial_program_correlations: Optional[List[np.array]] = None
    program_correlation_ids: Optional[List[ProgramId]] = None
    
def evaluate_program_in_environment(
        curiosity_program, rollout_timestep_pruning_hook_fn, early_termination_batch_data, 
        selected_index, tsp_params, extra_curiosity_programs, 
        extra_args, result_pipe_connection, fail_on_program_error=False):

    num_gpus, reward_combiner_program, programs_evaluation_folder = extra_args

    simulate_params = SearchExperimentList[
        tsp_params.SEARCH_PROGRAMS_EXPERIMENT_ID
    ]

    with tsp_params:
        with simulate_params:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print("selected index", selected_index, "num gpus", num_gpus, "selected gpu", selected_index % num_gpus)
                device = selected_index % num_gpus
            with mlca.helpers.config.DefaultDevice(device):

                trials_rollouts_mean_reward: TrialList[RolloutList[Reward]] = []
                trials_rollouts_episode_lengths: TrialList[RolloutList[EpisodeList[int]]] = []
                trials_rollouts_episode_end_rewards: TrialList[RolloutList[EpisodeList[Reward]]] = []
                trial_mean_episode_end_reward_at_timestep: Dict[Timestep, List[Reward]] = {}
                start_time = time.time()

                logger = mlca.helpers.debug.InMemoryLogger()
                profiler = mlca.helpers.debug.Profiler(logger, False)

                execution_had_error = False
                error = None
                early_terminated = False

                if extra_curiosity_programs is not None:
                    full_curiosity_program_list = [curiosity_program] + extra_curiosity_programs
                    trial_program_correlations: Optional[List[np.array]] = []
                    program_correlation_ids = [
                        p.program_id for p in full_curiosity_program_list
                    ]
                else:
                    trial_program_correlations = None
                    program_correlation_ids = None

                for trial_i in range(TspParams.current().NUM_TRIALS_PER_PROGRAM):
                    # Only log to tensorboard if this program has a special name.
                    log_tensorboard = curiosity_program.name is not None
                    if log_tensorboard:
                        tensorboard_writer = SummaryWriter(
                            log_dir='tensorboard/' + TspParams.current()._experiment_id + "_" + 
                            str(curiosity_program.program_id) + "_" + str(trial_i)
                        )
                    else:
                        tensorboard_writer = None
                    tensorboard_logger = mlca.helpers.debug.Logger(
                        tensorboard_writer)

                    # Note that we need to set the random seed directly before generating
                    # the datastructures as this determines the random weights we start with.
                    random_seed = 3 + 100 * trial_i + TspParams.current().RANDOM_SEED_OFFSET
                    print("Random seed", trial_i, random_seed)
                    mlca.helpers.util.set_random_seed(random_seed)

                    envs = make_envs(
                        curiosity_program, reward_combiner_program, random_seed
                    )

                    # Load model
                    # Note that we need to set the random seed directly before generating
                    # the datastructures as this determines the random weights we start with.
                    mlca.helpers.util.set_random_seed(random_seed + 1)

                    # Create the agent
                    actor_critic = a2c_ppo_acktr.model.Policy(
                        envs.observation_space.shape,
                        envs.action_space,
                        device,
                        base_kwargs={'recurrent': TspParams.current().AGENT_RECURRENT}).to(device)

                    agent = a2c_ppo_acktr.algo.PPO(
                        actor_critic,
                        TspParams.current().PPO_NEW_ARGS["clip_param"],
                        TspParams.current().PPO_NEW_ARGS["ppo_epoch"],
                        TspParams.current().PPO_NEW_ARGS["num_mini_batch"],
                        TspParams.current().PPO_NEW_ARGS["value_loss_coef"],
                        TspParams.current().PPO_NEW_ARGS["entropy_coef"],
                        lr=TspParams.current().PPO_NEW_ARGS["lr"],
                        eps=TspParams.current().PPO_NEW_ARGS["eps"],
                        max_grad_norm=TspParams.current().PPO_NEW_ARGS["max_grad_norm"])


                    internal_reward_module = make_reward_combiner_module(
                        curiosity_program, reward_combiner_program, envs, actor_critic)

                    if extra_curiosity_programs is not None:
                        extra_reward_modules = [
                            make_reward_combiner_module(
                                extra_curiosity_program, reward_combiner_program, envs, actor_critic)
                            for extra_curiosity_program in extra_curiosity_programs
                        ]
                        # List of 
                        #   [For each rollout, list of
                        #       [ For each reward module
                        #           [For each For each timestep, the reward]]]

                        full_reward_histories: List[List[List[Reward]]] = [
                            [ [] for reward_module in full_curiosity_program_list]
                            for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)
                        ] 
                    else:
                        extra_reward_modules = None
                    # Store data about our rollouts
                    rollouts = a2c_ppo_acktr.storage.RolloutStorage(
                        TspParams.current().PPO_FRAMES_PER_PROC,
                        TspParams.current().NUM_ROLLOUTS_PER_TRIAL,
                        envs.observation_space.shape, 
                        envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

                    # Setup reward tracking
                    episodes_final_rewards: List[List[Reward]] = [[] for _ in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                    episodes_lengths: List[List[int]] = [[] for _ in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                    cur_episode_sum_reward = [0 for _ in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                    cur_episode_length = [0 for _ in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]


                    # Find the first observation
                    obs = envs.reset()
                    rollouts.obs[0].copy_(obs)
                    rollouts.to(device)

                    profiler.tick(0, "Start trial")

                    # Train model
                    try:
                        start = time.time()
                        num_updates = int(
                            TspParams.current().STEPS_PER_ROLLOUT) // TspParams.current().PPO_FRAMES_PER_PROC # // TspParams.current().NUM_ROLLOUTS_PER_TRIAL
                                                
                        # Run the environment simulation, in batches, for a total of `num_updates` batches
                        cur_start_timestep = 0
                        for j in tqdm(range(num_updates)):
                            if TspParams.current().PPO_NEW_ARGS["use_linear_lr_decay"]:
                                # Decrease PPO learning rate linearly
                                a2c_ppo_acktr.utils.update_linear_schedule(
                                    agent.optimizer, j, num_updates,
                                    TspParams.current().PPO_NEW_ARGS["lr"])

                            profiler.tick(j, "PPO: Update LR")

                            for step in range(TspParams.current().PPO_FRAMES_PER_PROC):
                                cur_timestep = cur_start_timestep + step

                                # Sample actions
                                with torch.no_grad():
                                    # rollouts.obs[step] is the step we see on this step, because on step t it stores the new state at position t + 1 (and wraps around)
                                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                                        rollouts.obs[step], 
                                        rollouts.recurrent_hidden_states[step],
                                        rollouts.masks[step])

                                # Interact in all rollout environments simultaneously
                                obs, reward, done, infos = envs.step(action)

                                # Reward tracking
                                for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL):
                                    if done[rollout]:
                                        # Compute the reward of the rollout 
                                        if "MiniGrid" in TspParams.current().ENVIRONMENT:
                                            all_visited_states: Set[Any] = set().union(
                                                *[e.unwrapped.states_visited for e in envs.envs]
                                            )
                                            final_reward = len(all_visited_states)
                                        else:
                                            final_reward = cur_episode_sum_reward[rollout]

                                        episodes_final_rewards[rollout].append(final_reward)
                                        episodes_lengths[rollout].append(cur_episode_length[rollout])

                                        # Log the finished episode to tensorboard
                                        if log_tensorboard:
                                            tensorboard_logger.add_scalar(f'episode_final_reward_{trial_i}_{rollout}', final_reward, cur_timestep)
                                            tensorboard_logger.add_scalar(f'episode_final_length_{trial_i}_{rollout}', cur_episode_length[rollout], cur_timestep)
                                            tensorboard_logger.add_scalar(f"num_episodes_finished_{trial_i}_{rollout}", len(episodes_lengths[rollout]), cur_timestep)

                                        # Reset the episode trackers
                                        cur_episode_sum_reward[rollout] = 0
                                        cur_episode_length[rollout] = 0

                                    # Record the reward we got on this timestep
                                    cur_episode_sum_reward[rollout] += reward[rollout].cpu().item()
                                    cur_episode_length[rollout] += 1

                                # Log the finished episode to tensorboard
                                if log_tensorboard:
                                    mean_episode_end_reward = np.array(
                                        [np.array(episodes_final_rewards[rollout]).mean()
                                        for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                                    ).mean()
                                    mean_last_episode_end_reward = np.array(
                                        [episodes_final_rewards[rollout][-1] if len(episodes_final_rewards[rollout]) > 0 else math.nan
                                        for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                                    ).mean()
                                    if not np.isnan(mean_episode_end_reward):
                                        tensorboard_logger.add_scalar(f'mean_episode_end_reward_{trial_i}', mean_episode_end_reward, cur_timestep)
                                    if not np.isnan(mean_last_episode_end_reward):
                                        tensorboard_logger.add_scalar(f'mean_last_episode_end_reward_{trial_i}', mean_last_episode_end_reward, cur_timestep)

                                # Store this experience in our rollout buffer
                                # If done an episode, then clean the history of observations.
                                masks = torch.FloatTensor(
                                    [[0.0] if done_ else [1.0] for done_ in done])
                                bad_masks = torch.FloatTensor(
                                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                                    for info in infos])
                                rollouts.insert(obs, recurrent_hidden_states, action,
                                                action_log_prob, value, reward, masks, bad_masks)
                            profiler.tick(j, "PPO: Forward")

                            # Compute the intrinsic reward and the combined (remapped) rewards
                            remapped_rewards = internal_reward_module.calc_remapped_rewards(
                                rollouts, profiler, cur_start_timestep, tensorboard_logger, trial_i
                            )
                            if extra_reward_modules is not None:
                                remapped_extra_rewards = [
                                    extra_reward_module.calc_remapped_rewards(
                                        rollouts, profiler, cur_start_timestep, tensorboard_logger, trial_i
                                    )
                                    for extra_reward_module in extra_reward_modules
                                ]
                                for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL):
                                    for timestep in range(TspParams.current().PPO_FRAMES_PER_PROC):
                                        for curiosity_program_id, rewards in enumerate([remapped_rewards] + remapped_extra_rewards):
                                            full_reward_histories[rollout][curiosity_program_id].append(
                                                rewards[timestep][rollout].item()
                                            )
                                # print(full_reward_histories)

                            profiler.tick(j, "PPO: Remap rewards")

                            is_timestep_to_prune_on = SearchParams.current().ENABLE_EARLY_TERMINATION and \
                                cur_start_timestep % SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY == 0 and \
                                cur_start_timestep != 0
                            assert not early_termination_batch_data or SearchParams.current().EARLY_TERMINATION_CHECKING_FREQUENCY % TspParams.current().PPO_FRAMES_PER_PROC == 0
                            if is_timestep_to_prune_on:
                                if "MiniGrid" in TspParams.current().ENVIRONMENT:
                                    # We take the max here because we want the total # of states visited so far for this environment.
                                    mean_episode_end_reward = np.array(
                                        [np.array(episodes_final_rewards[rollout]).max() 
                                            if len(episodes_final_rewards[rollout]) > 0 else math.nan
                                        for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                                    ).max()
                                else:
                                    mean_episode_end_reward = np.array(
                                        [np.array(episodes_final_rewards[rollout]).mean()
                                        for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)]
                                    ).mean()

                                if not math.isnan(mean_episode_end_reward):
                                    if cur_start_timestep not in trial_mean_episode_end_reward_at_timestep:
                                        trial_mean_episode_end_reward_at_timestep[cur_start_timestep] = []
                                    trial_mean_episode_end_reward_at_timestep[cur_start_timestep].append(
                                        mean_episode_end_reward
                                    )

                                    prune = rollout_timestep_pruning_hook_fn and rollout_timestep_pruning_hook_fn(
                                        trial_i, 
                                        cur_start_timestep, 
                                        np.array(trial_mean_episode_end_reward_at_timestep[cur_start_timestep]).mean(), 
                                        early_termination_batch_data)

                                    if prune:
                                        print("Prune", trial_i, cur_start_timestep, np.array(trial_mean_episode_end_reward_at_timestep[cur_start_timestep]).mean())
                                        early_terminated = True
                                        break

                            profiler.tick(j, "PPO: Check pruning")

                            with torch.no_grad():
                                next_value = actor_critic.get_value(
                                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                    rollouts.masks[-1]).detach()
                            profiler.tick(j, "PPO: Get value")

                            rollouts.compute_returns(
                                next_value, 
                                TspParams.current().PPO_NEW_ARGS["use_gae"], 
                                TspParams.current().DECAY_RATE,
                                TspParams.current().PPO_NEW_ARGS["gae_lambda"],
                                TspParams.current().PPO_NEW_ARGS["use_proper_time_limits"])
                            profiler.tick(j, "PPO: Compute returns")

                            value_loss, action_loss, dist_entropy = agent.update(rollouts, profiler, j)
                            profiler.tick(j, "PPO: Update agent")

                            rollouts.after_update()
                            cur_start_timestep += TspParams.current().PPO_FRAMES_PER_PROC

                    except ProgramExecutionError as e:
                        if fail_on_program_error:
                            raise e
                        else:
                            execution_had_error = True
                            error = str(e)
                            print("execution_had_error")
                            print(e)
                            traceback.print_exc()
                            
                    except Exception as e:
                        if fail_on_program_error:
                            raise e
                        else:
                            print(e)
                            traceback.print_exc()
                            execution_had_error = True
                            error = str(e)

                    if "MiniGrid" in TspParams.current().ENVIRONMENT:
                        # Take the max here because we want the value from the last episode,
                        # which is the sum of all states visited so far
                        trials_rollouts_mean_reward.append(
                            [np.array(episodes_final_rewards[rollout]).max() if len(episodes_final_rewards[rollout]) > 0 else math.nan
                            for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)])
                    else:
                        trials_rollouts_mean_reward.append(
                            [np.array(episodes_final_rewards[rollout]).mean()
                            for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)])

                    trials_rollouts_episode_lengths.append(episodes_lengths)
                    trials_rollouts_episode_end_rewards.append(episodes_final_rewards)

                    if extra_curiosity_programs is not None:
                        num_programs = len(full_curiosity_program_list)
                        program_correlation=np.zeros((num_programs, num_programs))
                        for a, prog_a in enumerate(full_curiosity_program_list):
                            for b, prog_b in enumerate(full_curiosity_program_list):
                                if b > a:
                                    correlation = np.mean(np.array([
                                        scipy.stats.pearsonr(
                                            full_reward_histories[rollout][a], full_reward_histories[rollout][b]
                                        )[0]
                                        for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)
                                    ]))
                                    # print(correlation, full_reward_histories[rollout][a], full_reward_histories[rollout][b])
                                    program_correlation[a][b] = correlation
                                    program_correlation[b][a] = correlation
                        trial_program_correlations.append(program_correlation)
                    else: 
                        program_correlation = None

                    
                    if execution_had_error or early_terminated:
                        break

                result_data = ProgramTestResultData(
                    trials_rollouts_mean_reward,
                    trials_rollouts_episode_lengths,
                    trials_rollouts_episode_end_rewards,

                    execution_had_error,
                    error,
                    early_terminated,
                    early_termination_batch_data is not None,

                    selected_index,
                    time.time(),
                    start_time,
                    time.time() - start_time,
                    logger.avg_stats(),

                    curiosity_program.program_id if curiosity_program else None,
                    reward_combiner_program.program_id if reward_combiner_program else None,
                    device,

                    trial_program_correlations = trial_program_correlations,
                    program_correlation_ids = program_correlation_ids
                )

                if programs_evaluation_folder:
                    reward_combiner_program_i = reward_combiner_program.program_id if reward_combiner_program else None
                    with gzip.open(f"{programs_evaluation_folder}{curiosity_program.program_id}_{reward_combiner_program_i}.pickle.gz", 'wb') as output:
                        output.write(pickle.dumps(result_data))

                d = ProgramData(
                    curiosity_program.program_id, curiosity_program, reward_combiner_program, result_data,
                    _stats_for_program(result_data)
                )

                if result_pipe_connection:
                    print("Sending data!")
                    result_pipe_connection.send(d)

                return d            

def make_env():
    return gym.make(TspParams.current().ENVIRONMENT, **TspParams.current().ENVIRONMENT_KWARGS)

def make_envs(curiosity_program, reward_combiner_program, random_seed) -> Env:   
    mlca.helpers.util.set_random_seed(random_seed)

    return a2c_ppo_acktr.envs.make_vec_envs(
        TspParams.current().ENVIRONMENT, 
        random_seed, 
        TspParams.current().NUM_ROLLOUTS_PER_TRIAL,
        TspParams.current().DECAY_RATE,
        None, 
        device=mlca.helpers.config.DefaultDevice.current(),
        set_time_limit=TspParams.current().STEPS_PER_EPISODE,
        allow_early_resets=TspParams.current().ALLOW_EARLY_RESETS)

def make_reward_combiner_module(curiosity_program, reward_combiner_program, envs, policy) -> InternalRewardModule:   
    assert TspParams.current().BATCH_REWARD_COMPUTATION 

    curiosity_data_structure_values, curiosity_optimizer_values = curiosity_program.initialize_program_structures(
        make_env(), policy)
    if reward_combiner_program:
        reward_combiner_data_structure_values, reward_combiner_optimizer_values = reward_combiner_program.initialize_program_structures(
            make_env(), policy)
    else: 
        reward_combiner_data_structure_values, reward_combiner_optimizer_values = None, None

    return InternalRewardModule(
        curiosity_program, reward_combiner_program, 
        curiosity_data_structure_values, curiosity_optimizer_values, 
        reward_combiner_data_structure_values, reward_combiner_optimizer_values,
        envs, policy
    )
