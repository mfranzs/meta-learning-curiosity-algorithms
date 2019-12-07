"""
The internal reward module. Computes the internal reward on each state
given a rollout batch (and a intrinsic reward program), combines that internal 
reward with the external reward (using a reward combiner program), and
potentially normalizes it.
"""

import torch
import numpy as np
from typing import List

from baselines.common.running_mean_std import RunningMeanStd

from mlca.test_synthesized_programs_experiments import TspParams
import mlca.helpers.statistics.welfords_std
from mlca.helpers.config import DefaultDevice
from mlca.helpers.nn import one_hot

class InternalRewardModule:
    def __init__(self, curiosity_program, reward_combiner_program, 
            curiosity_data_structure_values, curiosity_optimizer_values,
            reward_combiner_data_structure_values, reward_combiner_optimizer_values,
            envs, policy):
        self.curiosity_program = curiosity_program
        self.reward_combiner_program = reward_combiner_program
        self.curiosity_data_structure_values = curiosity_data_structure_values
        self.curiosity_optimizer_values = curiosity_optimizer_values
        self.reward_combiner_data_structure_values =reward_combiner_data_structure_values
        self.reward_combiner_optimizer_values = reward_combiner_optimizer_values

        self.envs = envs

        self.internal_reward_normalizer_all = mlca.helpers.statistics.welfords_std.Welford()
        self.internal_reward_normalizer_window: List[int] = []
        
        # From https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py

        self.ret_rms = RunningMeanStd(shape=())
        self.clipob = 10.
        self.cliprew = 10.
        self.ret = np.zeros(TspParams.current().NUM_ROLLOUTS_PER_TRIAL)
        self.gamma = TspParams.current().DECAY_RATE
        assert self.gamma == .99
        self.epsilon = 1e-8

    def calc_remapped_rewards(self, rollouts, profiler, cur_start_timestep, tensorboard_logger, trial_i):
        states, prev_states, actions, extrinsic_rewards, normalized_timesteps, dones = \
            self.extract_from_rollout_buffer(rollouts, cur_start_timestep)

        STEPS_PER_CURIOSITY_UPDATE = TspParams.current().STEPS_PER_CURIOSITY_UPDATE * TspParams.current().NUM_ROLLOUTS_PER_TRIAL
        assert TspParams.current().PPO_FRAMES_PER_PROC * TspParams.current().NUM_ROLLOUTS_PER_TRIAL % STEPS_PER_CURIOSITY_UPDATE == 0
        remapped_rewards = []
        for internal_reward_update_batch in range(
                0, 
                TspParams.current().PPO_FRAMES_PER_PROC * TspParams.current().NUM_ROLLOUTS_PER_TRIAL,
                STEPS_PER_CURIOSITY_UPDATE):
            remapped_rewards.append(self.calc(
                prev_states[internal_reward_update_batch: internal_reward_update_batch + STEPS_PER_CURIOSITY_UPDATE],
                actions[internal_reward_update_batch: internal_reward_update_batch + STEPS_PER_CURIOSITY_UPDATE], 
                states[internal_reward_update_batch: internal_reward_update_batch + STEPS_PER_CURIOSITY_UPDATE], 
                dones[internal_reward_update_batch: internal_reward_update_batch + STEPS_PER_CURIOSITY_UPDATE],
                extrinsic_rewards[internal_reward_update_batch: internal_reward_update_batch + STEPS_PER_CURIOSITY_UPDATE], 
                normalized_timesteps[internal_reward_update_batch: internal_reward_update_batch + STEPS_PER_CURIOSITY_UPDATE],
                profiler, 
                cur_start_timestep, 
                tensorboard_logger=tensorboard_logger, 
                cur_start_timestep=cur_start_timestep + internal_reward_update_batch, 
                trial=trial_i
            ).view(
                TspParams.current().STEPS_PER_CURIOSITY_UPDATE, 
                # TspParams.current().PPO_FRAMES_PER_PROC),
                TspParams.current().NUM_ROLLOUTS_PER_TRIAL,
            ).unsqueeze(2))

        remapped_rewards_tensor = torch.cat(remapped_rewards)
        assert remapped_rewards_tensor.shape == rollouts.rewards.shape, (remapped_rewards_tensor.shape, rollouts.rewards.shape)
        rollouts.rewards = remapped_rewards_tensor

        return remapped_rewards_tensor


    def remap_actions(self, envs, actions):
        if not TspParams.current().CONTINUOUS_ACTION_SPACE:
            return one_hot(torch.cat(actions), self.envs.action_space.n).to(DefaultDevice.current())
        else:
            return torch.stack(actions).to(DefaultDevice.current())

    def extract_from_rollout_buffer(self, rollouts, cur_start_timestep):
        assert TspParams.current().REAL_BATCH_REWARD_COMPUTATION
        states = []
        prev_states = []
        actions = []
        extrinsic_rewards = []
        normalized_timesteps = []
        dones = []
        for timestep_i in range(TspParams.current().PPO_FRAMES_PER_PROC):
            for rollout in range(TspParams.current().NUM_ROLLOUTS_PER_TRIAL):
                i_episode = cur_start_timestep + timestep_i

                states.append(
                    rollouts.obs[timestep_i + 1][rollout])
                prev_states.append(
                    rollouts.obs[timestep_i][rollout])
                actions.append(
                    rollouts.actions[timestep_i][rollout])
                extrinsic_rewards.append(
                    rollouts.rewards[timestep_i][rollout].detach())
                dones.append(not rollouts.masks[timestep_i][rollout])
                normalized_timesteps.append(
                    i_episode / TspParams.current().STEPS_PER_ROLLOUT)

        states_tensor = torch.stack(states)
        prev_states_tensor = torch.stack(prev_states)
        actions_tensor = self.remap_actions(self.envs, actions)
        extrinsic_rewards_tensor = torch.cat(
            extrinsic_rewards)
        normalized_timesteps_tensor = torch.tensor(
            normalized_timesteps, dtype=torch.float, device=DefaultDevice.current())

        return states_tensor, prev_states_tensor, actions_tensor, extrinsic_rewards_tensor, normalized_timesteps_tensor, dones

    def calc(
            self, state, action, next_state, dones, extrinsic_reward,
            normalized_timestep, profiler=None, i_episode=None, tensorboard_logger=None, cur_start_timestep=None, trial=None):
        if TspParams.current().ONLY_EXTERNAL_REWARD:
            print("Only external")
            return extrinsic_reward
        else:
            intrinsic_reward = self._internal_reward(
                state, action, next_state, 
                profiler=None, i_episode=None)


            combined_reward = self._combined_reward(
                intrinsic_reward, extrinsic_reward, normalized_timestep)

            if TspParams.current().NORMALIZE_COMBINED_REWARD:
                assert not TspParams.current().PPO_NEW_ARGS["vec_normalize"]
                normalized_combined_reward = self._normalize_combined_reward(
                    combined_reward, dones
                )
            else:
                normalized_combined_reward = combined_reward


            if tensorboard_logger is not None and tensorboard_logger.tensorboard_writer is not None:
                i = 0

                if TspParams.current().REAL_BATCH_REWARD_COMPUTATION:
                    batch_num_timesteps = TspParams.current().STEPS_PER_CURIOSITY_UPDATE
                    #", TspParams.current().PPO_FRAMES_PER_PROC)
                else:
                    batch_num_timesteps = 1

                if TspParams.current().SHARE_CURIOSITY_MODULE_IN_TRIAL:
                    batch_num_rollouts = TspParams.current().NUM_ROLLOUTS_PER_TRIAL
                else:
                    batch_num_rollouts = 1

                # for timestep_i in range(batch_num_timesteps):
                #         print("intrinsic_reward", intrinsic_reward)
                #         tensorboard_logger.add_scalar(f'intrinsic_reward_{trial}_{rollout}', intrinsic_reward[i].cpu().item(), cur_start_timestep + timestep_i)
                #         tensorboard_logger.add_scalar(f'extrinsic_reward_{trial}_{rollout}', extrinsic_reward[i].cpu().item(), cur_start_timestep + timestep_i)
                #         tensorboard_logger.add_scalar(f'combined_reward_{trial}_{rollout}', combined_reward[i].cpu().item(), cur_start_timestep + timestep_i)
                #         tensorboard_logger.add_scalar(f'normalized_combined_reward{trial}_{rollout}', normalized_combined_reward[i].cpu().item(), cur_start_timestep + timestep_i)
                #         tensorboard_logger.add_scalar(f'normalized_timestep_{trial}_{rollout}', normalized_timestep[i].cpu().item(), cur_start_timestep + timestep_i)
                #         tensorboard_logger.add_scalar(f'done_{trial}_{rollout}', dones[i], cur_start_timestep + timestep_i)
                        
                #         i += 1
            
            return normalized_combined_reward

    def _normalize_combined_reward(self, combined_reward, dones):
        combined_reward = combined_reward.cpu().numpy()
        all_rews = []
        for timestep_i in range(TspParams.current().STEPS_PER_CURIOSITY_UPDATE):
            # ", TspParams.current().PPO_FRAMES_PER_PROC)):
            step_combined_reward = combined_reward[
                                timestep_i * TspParams.current().NUM_ROLLOUTS_PER_TRIAL:
                                (timestep_i + 1) * TspParams.current().NUM_ROLLOUTS_PER_TRIAL
                            ]
            
            self.ret = self.ret * self.gamma + step_combined_reward

            self.ret_rms.update(self.ret)
            rews = np.clip(step_combined_reward / np.sqrt(self.ret_rms.var +
                                                    self.epsilon), -self.cliprew, self.cliprew)

            timestep_dones = dones[
                timestep_i * TspParams.current().NUM_ROLLOUTS_PER_TRIAL:
                (timestep_i + 1) * TspParams.current().NUM_ROLLOUTS_PER_TRIAL
            ]
            # print(len(timestep_dones), timestep_dones, len(dones))
            # print(self.ret.shape)
            self.ret[timestep_dones] = 0.
            all_rews.append(torch.tensor(rews, device=DefaultDevice.current())) 
        
        return torch.cat(all_rews)

    def _internal_reward(
            self, state, action, next_state,
            profiler=None, i_episode=None):

        if state is None:
            return torch.zeros(action.shape[0], dtype=torch.float, device=DefaultDevice.current()) 

        input_values = {
            "observation_image": state,
            "action_one_hot": action,
            "new_observation_image":  next_state,
        }

        assert set(input_values.keys()) == set(
            i.name for i in self.curiosity_program.input_variables), ("available values", set(input_values.keys()), "requested values", set(
                i.name for i in self.curiosity_program.input_variables))

        input_values_by_variable = {
            i: input_values[i.name] for i in self.curiosity_program.input_variables
        }

        r = self.curiosity_program.execute(
            input_values_by_variable,
            self.curiosity_data_structure_values,
            self.curiosity_optimizer_values,
            profiler=profiler,
            print_on_error=False,
            i_episode=i_episode
        ).detach()

        return self._normalize_internal_reward(r)

    def _normalize_internal_reward(self, r):
        if TspParams.current().NORMALIZE_INTERNAL_REWARD == "ALL":
            for k in r.detach():
                self.internal_reward_normalizer_all.update(k)

            std = self.internal_reward_normalizer_all.std
            if std == 0:
                return torch.sign(r)
            else:
                return  r / std
        elif TspParams.current().NORMALIZE_INTERNAL_REWARD == "100":
            REWARD_WINDOW = 5 * 100
            for k in r.detach():
                self.internal_reward_normalizer_window.append(k.item())
            self.internal_reward_normalizer_window = self.internal_reward_normalizer_window[-REWARD_WINDOW:]
            std = torch.tensor(self.internal_reward_normalizer_window, device=DefaultDevice.current()).std()
            # print(r, std, r/std)
            # print(r[0], std, r[0]/std)
            if torch.isnan(std):
                return torch.sign(r)
            else:
                return r / std
        else:
            return r.detach()

    def _normalize_external_reward(self, external_reward):
        if TspParams.current().NORMALIZE_EXTERNAL_REWARD == "MANUAL":
            return external_reward / TspParams.current().NORMALIZE_EXTERNAL_REWARD_MANUAL_LEVEL
        else:
            return external_reward

    def _combined_reward(
            self, intrinsic_reward, raw_extrinsic_reward,
            normalized_timestep):

        extrinsic_reward = self._normalize_external_reward(raw_extrinsic_reward)
        
        if self.reward_combiner_program is None:
            # print("Missing reward combiner")
            return intrinsic_reward + extrinsic_reward
        else:
            input_values = {
                "intrinsic_reward": intrinsic_reward,
                "extrinsic_reward": extrinsic_reward,
                "normalized_timestep": normalized_timestep,
            }
            assert type(intrinsic_reward) == torch.Tensor and type(extrinsic_reward) == torch.Tensor and type(
                normalized_timestep) == torch.Tensor, (type(intrinsic_reward), type(extrinsic_reward), type(normalized_timestep))

            assert set(input_values.keys()) == set(
                i.name for i in self.reward_combiner_program.input_variables), ("available values", set(input_values.keys()), "requested values", set(
                    i.name for i in self.reward_combiner_program.input_variables))

            input_values_by_variable = {
                i: input_values[i.name] for i in self.reward_combiner_program.input_variables
            }

            r = self.reward_combiner_program.execute(
                input_values_by_variable,
                self.reward_combiner_data_structure_values,
                self.reward_combiner_optimizer_values,
                print_on_error=False
            )

            return r.detach()
