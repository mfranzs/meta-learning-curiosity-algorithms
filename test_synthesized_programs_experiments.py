from mlca.helpers.experiment_params import ExperimentParameters, ExperimentParameterList
from enum import Enum
from typing import Optional, List, Any
from dataclasses import dataclass

"""TestSynthesizedProgramsExperimentParameters"""
@dataclass
class TspParams(ExperimentParameters):
    class ExperimentType(Enum):
        CURIOSITY_SEARCH = 1
        REWARD_COMBINER_SEARCH = 2

    ENVIRONMENT: str
    ENVIRONMENT_KWARGS: dict
    CONTINUOUS_ACTION_SPACE: bool
    RANDOM_SEED_OFFSET: int

    NUM_TRIALS_PER_PROGRAM: int
    NUM_ROLLOUTS_PER_TRIAL: int
    STEPS_PER_EPISODE: int
    STEPS_PER_ROLLOUT: int

    EXPERIMENT_TYPE: ExperimentType

    AGENT_RECURRENT: bool
    STEPS_PER_CURIOSITY_UPDATE: Optional[int]

    CURIOSITY_PROGRAMS_NAME: str
    
    MAX_VARIABLE_BUFFER_SIZE: int
    KNN_BUFFER_SIZE_SMALL: int
    KNN_BUFFER_SIZE_LARGE: int

    FIXED_CURIOSITY_PROGRAM_ID: Optional[int]

    FIXED_REWARD_COMBINER_PROGRAM_ID: Optional[int]

    DECAY_RATE: float
    ALLOW_EARLY_RESETS: bool
    BATCH_REWARD_COMPUTATION: bool
    REAL_BATCH_REWARD_COMPUTATION: bool
    PPO_FRAMES_PER_PROC: int

    NORMALIZE_COMBINED_REWARD: bool

    SEARCH_PROGRAMS_EXPERIMENT_ID: str

    SHARE_CURIOSITY_MODULE_IN_TRIAL: bool
    
    RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE: Optional[List[int]]

    NORMALIZE_COMBINER_REWARD: bool

    LEARNING_RATE: float

    PPO_NEW_ARGS: Any

    SPLIT_ACROSS_MACHINES: Optional[int] = None
    BATCH_SIZE: Optional[int] = None
    TARGET_UPDATE: Optional[int] = None
    MAX_EPISODES: Optional[int] = None
    WEIRD_EPS_DECAY: Optional[int] = None
    EPS_DECAY: Optional[int] = None
    EPS_START: Optional[int] = None
    EPS_END: Optional[int] = None
    GAMMA: Optional[int] = None
    ADD_V_LOSS: Optional[bool] = None
    REWARD_COMBINER_PROGRAMS_NAME: Optional[str] = None
    ONLY_EXTERNAL_REWARD: Optional[bool] = False
    NORMALIZE_INTERNAL_REWARD: Optional[str] = None
    NORMALIZE_EXTERNAL_REWARD: Optional[str] = None

    COMPUTE_PROGRAM_CORRELATIONS: bool = False
    COMPUTE_PROGRAM_CORRELATIONS_PROGRAMS_PER_BATCH: Optional[int] = None

    FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS : bool = False


TspExperimentList = ExperimentParameterList()

TspExperimentList["2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"] = TspParams(
    RANDOM_SEED_OFFSET = 1,

    EXPERIMENT_TYPE = TspParams.ExperimentType.CURIOSITY_SEARCH,
    SEARCH_PROGRAMS_EXPERIMENT_ID = "ss-1-knn-10-fv-1-pairs-early-termination-2-62",
    
    REWARD_COMBINER_PROGRAMS_NAME = "programs_reward_combiner_3_v4",    
    CURIOSITY_PROGRAMS_NAME = "programs_curiosity_7_v7",

    NUM_ROLLOUTS_PER_TRIAL = 5,
    STEPS_PER_EPISODE = 500,
    STEPS_PER_ROLLOUT = 2500,    
    STEPS_PER_CURIOSITY_UPDATE = 1,

    AGENT_RECURRENT = False,

    SHARE_CURIOSITY_MODULE_IN_TRIAL = True,
    REAL_BATCH_REWARD_COMPUTATION = True,

    RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = None,
    NORMALIZE_COMBINER_REWARD = True,

    MAX_VARIABLE_BUFFER_SIZE = 1000,
    KNN_BUFFER_SIZE_SMALL = 100,
    KNN_BUFFER_SIZE_LARGE = 1000,

    FIXED_CURIOSITY_PROGRAM_ID = None,
    FIXED_REWARD_COMBINER_PROGRAM_ID = 231,

    DECAY_RATE = 0.99,
    ALLOW_EARLY_RESETS = True,
    BATCH_REWARD_COMPUTATION = True,
    PPO_FRAMES_PER_PROC = 12,

    ENVIRONMENT = "MiniGridEmptyEnv30x30Unwrapped-v0",
    ENVIRONMENT_KWARGS = {},
    CONTINUOUS_ACTION_SPACE = False,
    NUM_TRIALS_PER_PROGRAM = 5,

    NORMALIZE_COMBINED_REWARD = True,

    LEARNING_RATE = 0.005, 

    PPO_NEW_ARGS = {
        # From "Atari PPO" https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        "use_gae": True,
        "use_linear_lr_decay": True,
        "vec_normalize": False,
        "lr": 2.5e-4,
        "clip_param": 0.1,
        "value_loss_coef": 0.5,
        "num_mini_batch": 4,
        "log_interval": 1,
        "entropy_coef": 0,  # 0 so doesn't internally explore more , prev was 0.01,
        # from defaults https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master//a2c_ppo_acktr/arguments.py#L6:5
        "eps": 1e-5,
        "ppo_epoch": 4,
        "gae_lambda": 0.95,
        "max_grad_norm": .5,
        "use_proper_time_limits": False,
        # Other diff parameters:
        # num_steps":  128,
    }
)

TspExperimentList["temp-2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"] = \
    TspExperimentList["2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
        # REWARD_COMBINER_PROGRAMS_NAME = "programs_reward_combiner_3_v5",    
        # CURIOSITY_PROGRAMS_NAME = "programs_curiosity_4_v8",
        # FIXED_REWARD_COMBINER_PROGRAM_ID=None,
        PPO_FRAMES_PER_PROC = 3,
        STEPS_PER_EPISODE = 3,
        STEPS_PER_ROLLOUT = 12,
        NUM_TRIALS_PER_PROGRAM = 1
    )

TspExperimentList["2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials"] = \
    TspExperimentList["2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
        STEPS_PER_EPISODE=100,
        ENVIRONMENT = "MiniGridEmptyEnv15x15Unwrapped-v0",
    )

TspExperimentList["2-87_acrobot_new-ppo-real-batched-shared_5000-steps_5-trials"] = TspExperimentList["2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
    ENVIRONMENT = "Acrobot-v1",
    #OBSERVATION_ENCODING = None,

    STEPS_PER_ROLLOUT = 100 * 50,
    STEPS_PER_EPISODE = 100,

    NUM_ROLLOUTS_PER_TRIAL = 5,
    NUM_TRIALS_PER_PROGRAM = 5,

)

TspExperimentList["2-88_ant_new-ppo-external-only"] = TspExperimentList["2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
ENVIRONMENT = "Ant-v2",
CONTINUOUS_ACTION_SPACE = True,

STEPS_PER_ROLLOUT = 100000, #0,
STEPS_PER_EPISODE = None,
NUM_ROLLOUTS_PER_TRIAL = 1,
NUM_TRIALS_PER_PROGRAM = 2,
PPO_FRAMES_PER_PROC = 2048, 

NORMALIZE_COMBINED_REWARD = True, 

PPO_NEW_ARGS = {
        "use_gae":True,
    "use_linear_lr_decay":True,
    "vec_normalize":False, # WARNING THIS IS DIFF AS OF SAT 4PM
    "lr":3e-4,
    "clip_param":0.2,
    "value_loss_coef":0.5,
    "num_mini_batch":32,
    "entropy_coef":0, 
    "eps":1e-5,
    "ppo_epoch":10,
    "gae_lambda":0.95,
    "max_grad_norm":.5,
    "use_proper_time_limits":True,
}
)

TspExperimentList["2-88_ant_new-ppo"] = TspExperimentList["2-88_ant_new-ppo-external-only"].replace(
    ONLY_EXTERNAL_REWARD = False
)

TspExperimentList["2-88_ant_new-ppo-normalize"] = TspExperimentList["2-88_ant_new-ppo"].replace(
    NORMALIZE_INTERNAL_REWARD = "ALL",
)

TspExperimentList["2-89_lunar-lander_new-ppo"] = TspExperimentList["2-88_ant_new-ppo"].replace(
    ENVIRONMENT = "LunarLander-v2",
    CONTINUOUS_ACTION_SPACE = False,
    NUM_TRIALS_PER_PROGRAM = 2,
)

TspExperimentList["2-89_lunar-lander_new-ppo-external-only"] = TspExperimentList["2-89_lunar-lander_new-ppo"].replace(
    ONLY_EXTERNAL_REWARD = False
)

TspExperimentList["2-89_lunar-lander_new-ppo-normalize"] = TspExperimentList["2-89_lunar-lander_new-ppo"].replace(
    NORMALIZE_INTERNAL_REWARD = "ALL",
)

TspExperimentList["2-89_lunar-lander_new-ppo-more-curiosity"] = TspExperimentList["2-89_lunar-lander_new-ppo"].replace(
    STEPS_PER_CURIOSITY_UPDATE = 512,
    FIXED_REWARD_COMBINER_PROGRAM_ID = None,
)

TspExperimentList["2-90_acrobot_new-ppo"] = TspExperimentList["2-88_ant_new-ppo"].replace(
    ENVIRONMENT = "Acrobot-v1",
    CONTINUOUS_ACTION_SPACE = False,
    NUM_TRIALS_PER_PROGRAM = 2,
)

TspExperimentList["2-90_acrobot_new-ppo-normalize"] = TspExperimentList["2-90_acrobot_new-ppo"].replace(
    NORMALIZE_INTERNAL_REWARD = "ALL",
)

TspExperimentList["2-90_acrobot_new-ppo-nocombiner"] = TspExperimentList["2-90_acrobot_new-ppo"].replace(
        FIXED_REWARD_COMBINER_PROGRAM_ID =  None
)

TspExperimentList["2-90_acrobot_new-ppo-more-curiosity"] = TspExperimentList["2-90_acrobot_new-ppo"].replace(
    STEPS_PER_CURIOSITY_UPDATE = 512,
    FIXED_REWARD_COMBINER_PROGRAM_ID = None,
)

TspExperimentList["2-91_reacher_new-ppo"] = TspExperimentList["2-88_ant_new-ppo-external-only"].replace(
    ENVIRONMENT = "Reacher-v2",
    ONLY_EXTERNAL_REWARD = False,
    NUM_TRIALS_PER_PROGRAM = 2,
    STEPS_PER_ROLLOUT = 100000, #0,  ,
    )

TspExperimentList["2-91_reacher_new-ppo-normalize"] = TspExperimentList["2-91_reacher_new-ppo"].replace(
    NORMALIZE_INTERNAL_REWARD = "ALL",
)

TspExperimentList["2-91_reacher_new-ppo-more-curiosity"] = TspExperimentList["2-91_reacher_new-ppo"].replace(
    STEPS_PER_CURIOSITY_UPDATE = 512,
    FIXED_REWARD_COMBINER_PROGRAM_ID = None,
)

TspExperimentList["2-89_15x15_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination"] = TspExperimentList["2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
    SEARCH_PROGRAMS_EXPERIMENT_ID = "ss-1-knn-10-fv-1-pairs-early-termination-2-80",
    FIXED_REWARD_COMBINER_PROGRAM_ID = None,
)

TspExperimentList["2-89_15x15_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination-original"] = TspExperimentList["2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
    STEPS_PER_ROLLOUT = 500,
    NUM_ROLLOUTS_PER_TRIAL = 5,
    NUM_TRIALS_PER_PROGRAM = 10,
    STEPS_PER_EPISODE = 100,
)


TspExperimentList["2-92_15x15_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination"] = TspExperimentList["2-89_15x15_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination"].replace(
    SEARCH_PROGRAMS_EXPERIMENT_ID = "ss-1-knn-10-fv-1-pairs-early-termination-2-89",
    SPLIT_ACROSS_MACHINES = 4,
)

TspExperimentList["2-93_krazyworld_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination"] = TspExperimentList["2-92_15x15_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination"].replace(
    ENVIRONMENT = "KrazyWorld-v0",
    SPLIT_ACROSS_MACHINES = None,
)

# Full gridworld search for ICLR!
TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = \
    TspExperimentList["2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
        SEARCH_PROGRAMS_EXPERIMENT_ID = 'ss-1-knn-10-fv-1-pairs-early-termination-2-96',

        SHARE_CURIOSITY_MODULE_IN_TRIAL = True,
        REAL_BATCH_REWARD_COMPUTATION = True,
        STEPS_PER_CURIOSITY_UPDATE = 1,
        STEPS_PER_ROLLOUT = 1000,

        SPLIT_ACROSS_MACHINES = 4,

        MAX_VARIABLE_BUFFER_SIZE = 100,
        KNN_BUFFER_SIZE_SMALL = 500,

        REWARD_COMBINER_PROGRAMS_NAME = None,
        FIXED_REWARD_COMBINER_PROGRAM_ID = None,
    )

TspExperimentList["2-96_regression-test"] = \
    TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(

    )

TspExperimentList["2-96_regression-test-2"] = \
    TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(

    )

TspExperimentList["2-96_program-correlation-5"] = \
    TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
        COMPUTE_PROGRAM_CORRELATIONS = True,
        COMPUTE_PROGRAM_CORRELATIONS_PROGRAMS_PER_BATCH = 5,
    )


# Krazyworld hyperparam search
TspExperimentList["2-97_krazyworld_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "KrazyWorld-v0",
)

# Enable reward combiner (Abstract) hyperparam search
TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = None ,

    REWARD_COMBINER_PROGRAMS_NAME = "programs_reward_combiner_3_v4",
    FIXED_REWARD_COMBINER_PROGRAM_ID = 231,

    NORMALIZE_COMBINED_REWARD = True,

    # Atari PPO
    PPO_NEW_ARGS = {
        # From "Atari PPO" https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        "use_gae":True,
        "use_linear_lr_decay":True,
        "vec_normalize":False,  
        "lr":2.5e-4,
        "clip_param":0.1,
        "value_loss_coef":0.5,
        "num_mini_batch":4,
        "log_interval":1,
        "entropy_coef":0,  # 0 so doesn't internally explore more , prev was 0.01,
        # from defaults https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master//a2c_ppo_acktr/arguments.py#L6:5
        "eps":1e-5,
        "ppo_epoch":4,
        "gae_lambda":0.95,
        "max_grad_norm":.5,
        "use_proper_time_limits":True, 
        # Other diff parameters:
        # num_steps":  128,
    }
)

# Enable reward combiner (Abstract) hyperparam search
TspExperimentList["2-97_enable_reward_combiner_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = None,

    REWARD_COMBINER_PROGRAMS_NAME = "programs_reward_combiner_3_v4",
    FIXED_REWARD_COMBINER_PROGRAM_ID = 231,

    NORMALIZE_COMBINED_REWARD = True,

    PPO_FRAMES_PER_PROC = 256,
    # Mujoco PPO
    PPO_NEW_ARGS = {
        "use_gae":True,
        "use_linear_lr_decay":True,
        "vec_normalize":False,
        "lr":3e-4,
        "clip_param":0.2,
        "value_loss_coef":0.5,
        "num_mini_batch":32,
        "entropy_coef":0,
        "eps":1e-5,
        "ppo_epoch":10,
        "gae_lambda":0.95,
        "max_grad_norm":.5,
        "use_proper_time_limits":True,
    }
)

# LL, LL CONTINUOUS, ACROBOT WITH ATARI PPO
# Lunarlander hyperparam search
TspExperimentList["2-97_lunarlander_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "LunarLander-v2",

    STEPS_PER_EPISODE = 500,
    STEPS_PER_ROLLOUT = 400 * 125,  # We previously saw average episode length of ~ 125
)

# Lunarlander hyperparam search
TspExperimentList["2-97_lunarlander-continuous_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "LunarLanderContinuous-v2",
    CONTINUOUS_ACTION_SPACE = True,

    STEPS_PER_EPISODE = 500,
    STEPS_PER_ROLLOUT = 400 * 125,  # We previously saw average episode length of ~ 125
)

# Acrobot hyperparam search
TspExperimentList["2-97_acrobot_new_atari-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 100 * 50,
    STEPS_PER_EPISODE = 100,
)

# LL, LL CONTINUOUS, ACROBOT WITH MUJOCO PPO
# Lunarlander hyperparam search
TspExperimentList["2-97_lunarlander_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "LunarLander-v2",

    STEPS_PER_EPISODE = 500,
    STEPS_PER_ROLLOUT = 400 * 125,  # We previously saw average episode length of ~ 125
)

# Lunarlander hyperparam search
TspExperimentList["2-97_lunarlander-continuous_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "LunarLanderContinuous-v2",
    CONTINUOUS_ACTION_SPACE = True,

    STEPS_PER_EPISODE = 500,
    STEPS_PER_ROLLOUT = 400 * 125,  # We previously saw average episode length of ~ 125
)

# Acrobot hyperparam search
TspExperimentList["2-97_acrobot_new_mujoco-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 100 * 50,
    STEPS_PER_EPISODE = 100,
)

# Fixing Acrobot w. more steps?


# Acrobot hyperparam search
TspExperimentList["2-97_acrobot_new_atari-ppo-real-batched-shared_25000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 500 * 50,
    STEPS_PER_EPISODE = 500,
)

TspExperimentList["2-97_acrobot_new_mujoco-ppo-real-batched-shared_25000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 500 * 50,
    STEPS_PER_EPISODE = 500,
)

TspExperimentList["2-97_acrobot_new_atari-ppo-real-batched-shared_125000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 500 * 500,
    STEPS_PER_EPISODE = 500,
)

TspExperimentList["2-97_acrobot_new_mujoco-ppo-real-batched-shared_125000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 500 * 500,
    STEPS_PER_EPISODE = 500,
)


# MUJOCO hyperparam search
TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = None,
CONTINUOUS_ACTION_SPACE = True,

REWARD_COMBINER_PROGRAMS_NAME = "programs_reward_combiner_3_v4",
FIXED_REWARD_COMBINER_PROGRAM_ID = 231,

STEPS_PER_ROLLOUT = 500000,
STEPS_PER_EPISODE = None,
NUM_ROLLOUTS_PER_TRIAL = 1,
NUM_TRIALS_PER_PROGRAM = 2,
PPO_FRAMES_PER_PROC = 2048, 

NORMALIZE_COMBINED_REWARD = True, 

PPO_NEW_ARGS = {
    "use_gae":True,
    "use_linear_lr_decay":True,
    "vec_normalize":False, # WARNING THIS IS DIFF AS OF SAT 4PM
    "lr":3e-4,
    "clip_param":0.2,
    "value_loss_coef":0.5,
    "num_mini_batch":32,
    "entropy_coef":0, 
    "eps":1e-5,
    "ppo_epoch":10,
    "gae_lambda":0.95,
    "max_grad_norm":.5,
    "use_proper_time_limits":True,
}
)

# Ant hyperparam search
TspExperimentList["2-97_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Ant-v2",
)

# Hopper hyperparam search
TspExperimentList["2-97_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Hopper-v2",
)

# Walker hyperparam search
TspExperimentList["2-97_walker_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Walker2d-v2",
)

# Acrobot ICLR search!
TspExperimentList["2-98_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_enable_reward_combiner_atari_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    ENVIRONMENT = "Acrobot-v1",

    STEPS_PER_ROLLOUT = 7500,
    STEPS_PER_EPISODE = 500,

    NUM_TRIALS_PER_PROGRAM = 2,

    SEARCH_PROGRAMS_EXPERIMENT_ID = "ss-1-knn-10-fv-1-pairs-early-termination-2-98",
    RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [18439, 25716, 26235, 32962, 35746, 35816, 37122, 42850, 43687, 43801, 50211, 50269, 50700, 51418, 51442, 2668, 16880, 19798, 25908, 31555, 34941, 40039, 43504, 51266, 9865, 10483, 10556, 13735, 16077, 16459, 16628, 20150, 27871, 39466, 43072, 3761, 7205, 8237, 11078, 11970, 14493, 15712, 16490, 17339, 19693, 19761, 24548, 25526, 27734, 30040, 30609, 31118, 34297, 35773, 36734, 42687, 47274, 48322, 50141, 2387, 6160, 6162, 14185, 16601, 19760, 20642, 20971, 25573, 26383, 34947, 37141, 45212, 49087, 49892, 14703, 15320, 16390, 16773, 22852, 22961, 23376, 24648, 28913, 32122, 37096, 37478, 44219, 49618, 50841, 50977, 656, 9790, 16509, 16710, 17002, 21740, 22458, 25247, 37538, 38548, 42915, 43511, 47685, 48454, 411, 4589, 16238, 17010, 19502, 19847, 25027, 28856, 32678, 33250, 34566, 37482, 39276, 40691, 41405, 42006, 43437, 43886, 19076, 20205, 20854, 38093, 40500, 40897, 43874, 45603, 49804, 50728, 9406, 9744, 11106, 14635, 15037, 19949, 28304, 28670, 30177, 33824, 34047, 34524, 35249, 37254, 38347, 39248, 43161, 43780, 51299, 8533, 19857, 23543, 28281, 31620, 32913, 34568, 34811, 35388, 38594, 40141, 42752, 48616, 51279, 12745, 19579, 19788, 37779, 47945, 1703, 14073, 28073, 35648, 35819, 38569, 50122, 50247, 51061, 9065, 16061, 16695, 26267, 31187, 31774, 32933, 35358, 40684, 43586, 44562, 47596, 49006, 7355, 16698, 20249, 20360, 23550, 25175, 28707, 29231, 31329, 32717, 40985, 42996, 43643, 47602, 50780, 81, 1045, 10907, 16587, 17918, 22462, 22467, 33895, 34007, 34104, 34559, 38606, 42132, 42953, 44081, 44248, 44673, 45483, 45638, 7159, 7438, 10335, 10701, 11094, 12492, 19334, 19845, 28992, 29222, 33498, 36096, 41521, 47263, 10194, 11320, 20334, 20968, 21948, 22502, 26098, 37558, 40476, 44302, 44793, 49655, 22054, 34562, 36254, 42489, 50353, 163, 6352, 9819, 15313, 16040, 23217, 26442, 28588, 28717, 28788, 37468, 38322, 38567, 39510, 51309, 2836, 2894, 3734, 8084, 9480, 16633, 25952, 26444, 26739, 39957, 40637, 43159, 44173, 49713, 2131, 6497, 11711, 14702, 20431, 21362, 25249, 28067, 29999, 31942, 32487, 34761, 37762, 38575, 48310, 50223, 51137, 10513, 11665, 14938, 17382, 18735, 19706, 24994, 28753, 30878, 31304, 38182, 42898, 49258, 51518, 9690, 14672, 16108, 31888, 50016, 20809, 29344, 29693, 34509, 35903, 36188, 36359, 42460, 46421, 19, 15315, 19656, 20277, 25230, 31348, 32379, 39643, 45181, 46647, 48791, 8376, 16921, 19183, 19480, 20993, 23685, 25039, 28758, 32195, 33768, 34001, 36729, 37454, 49817, 51569, 5951, 11038, 20228, 20817, 22887, 22958, 26334, 27812, 28653, 30975, 32533, 32689, 38343, 41745, 42940, 49860, 3168, 7457, 9551, 22642, 27632, 28639, 31852, 32125, 40401, 41570, 41898, 7582, 14430, 20594, 22076, 25177, 26195, 27888, 32515, 34849, 39611, 40973, 41374, 43180, 48253, 21688, 34788, 477, 10623, 17477, 20875, 27852, 37570, 44179, 47487, 11505, 12524, 20137, 31200, 32488, 39368, 44427, 46989, 49346, 1125, 16466, 22570, 26323, 35149, 36224, 37076, 37399, 38219, 38654, 41513, 42554, 42768, 49211, 50593, 50885, 51803, 129, 185, 5959, 12383, 19287, 22047, 26042, 27643, 29206, 37145, 38472, 40049, 40359, 41389, 41731, 41976, 45566, 310, 12489, 13783, 15666, 17304, 17576, 23051, 27158, 30810, 35133, 37024, 41268, 43881, 48732, 29901, 38386, 41629, 50196, 176, 2056, 15436, 23399, 25203, 28678, 33983, 34696, 37502, 39733, 41006, 41920, 942, 13017, 13582, 13611, 16295, 19878, 40780, 43374, 47525, 51229, 1943, 6564, 9618, 14641, 16342, 20706, 28075, 31241, 34540, 44007, 47396, 49457, 50563, 11164, 19978, 22289, 22317, 27854, 33204, 35306, 43425, 29016, 35329, 21708, 24288, 28433, 34547, 38550, 43042, 43757, 49307, 9941, 9943, 13337, 13479, 15702, 27066, 32302, 33669, 35450, 50673, 3951, 11933, 14465, 17375, 20475, 20860, 23518, 25158, 25289, 26162, 35264, 40069, 40745, 41478, 47901, 49294, 8273, 9687, 11202, 15293, 17102, 21396, 22838, 26526, 29242, 33940, 35976, 42832, 45614, 51689, 16298, 24521, 26510, 27717, 35173, 40126, 40988, 43053, 5052, 15951, 25375, 27875, 33680, 33718, 34532, 37445, 38453, 42403, 45357, 48418, 48619, 24030, 34622, 34665, 50699, 450, 1561, 15894, 20207, 20773, 23280, 23686, 29285, 32298, 34805, 38319, 39639, 16658, 23689, 31045, 32387, 33497, 49904, 10223, 11617, 15317, 28233, 33984, 36346, 37175, 38405, 38619, 41542, 45345, 49703, 50324, 462, 8090, 8637, 10379, 13345, 15295, 15663, 16654, 20974, 32291, 40301, 44339, 45540, 177, 992, 8665, 15084, 20313, 22447, 27763, 28743, 32078, 33111, 37161, 40707, 49880, 9554, 15234, 20332, 20856, 28272, 32009, 32491, 34442, 38413, 45509, 46680, 5561, 8516, 15947, 15973, 15978, 25463, 42331, 44180, 47369, 47529, 49184, 235, 4518, 9635, 12050, 24656, 25600, 28523, 28577, 31042, 36904, 38515, 41117, 46314, 47493, 48198, 50583, 924, 1467, 4381, 9131, 9275, 13499, 16812, 28325, 30652, 31001, 34453, 39309, 40243, 47284, 51110, 7258, 7934, 20090, 22552, 33173, 41652, 48599, 49260, 34053, 50052, 8895, 22430, 32990, 39641, 47935, 50513, 3425, 27531, 31393, 35178, 35447, 41037, 41046, 44212, 6362, 9356, 13529, 15451, 16711, 26529, 28649, 28918, 33668, 36306, 37106, 43235, 15319, 17065, 22270, 32474, 40127, 40579, 51103, 10337, 17021, 26512, 36948, 39581, 44114, 44837, 48211, 28452, 43110, 15946, 25144, 32393, 32534, 36616, 42183, 43867, 44729, 50915, 19455, 22988, 25481, 28316, 10860, 13623, 28410, 34783, 35666, 38028, 40138, 45607, 47331, 48520, 6214, 6625, 7620, 17300, 28255, 31069, 32790, 39302, 40786, 45193, 45337, 139, 5349, 26441, 27428, 33212, 38327, 43988, 50778, 28542, 34909, 37334, 42823, 50567, 19266, 37028, 41038, 41822, 43873, 45382, 10696, 17098, 17278, 17302, 22454, 23285, 27792, 35150, 37206, 43125, 43241, 51208, 51616, 14704, 23682, 32603, 32621, 34790, 35855, 36342, 41654, 44280, 4254, 8042, 31703, 40776, 40867, 43224, 15911, 15949, 37886, 39395, 40108, 46640, 49743, 49885, 50739, 6108, 28083, 28434, 28650, 36586, 6233, 10282, 14281, 17280, 29056, 33945, 38338, 44384, 48783, 51358, 10334, 21943, 27431, 41195, 3211, 12945, 22690, 26570, 31130, 31536, 32580, 34141, 37197, 39850, 43897, 47247, 8950, 15867, 19549, 22324, 22696, 50713, 6368, 21185, 22598, 33552, 39396, 39601, 43825, 10771, 15705, 15707, 16146, 37537, 38503, 17556, 29117, 33787, 33886, 35242, 22784, 37794, 8747, 9117, 11772, 40777, 40970, 46068, 50677, 12327, 27718, 31736, 38630, 40898, 2438, 10068, 27867, 34674, 36144, 38187, 49180, 50736, 1917, 11205, 18083, 20815, 50250, 50819, 27557, 34880, 35035, 36750, 38018, 39569, 49018, 50622, 7439, 9721, 26335, 28698, 32308, 40602, 41217, 41641, 47327, 48990, 13792, 19483, 19773, 22934, 34974, 35685, 46249, 6282, 6391, 15325, 25513, 37775, 41307, 43118, 33707, 44693, 49192, 15581, 7601, 16599, 26992, 27350, 14870, 27874, 49272, 8191, 15706, 17337, 23331, 33193, 34949, 49497, 51135, 5756, 15885, 24824, 36880, 20954, 23130, 24934, 47122, 7025, 28430, 43082, 44581, 16912, 24634, 38209, 42760, 45238, 4152, 14119, 20246, 28612, 31761, 32854, 50964, 995, 5843, 14692, 20196, 23211, 23622, 33862, 34798, 35041, 37016, 40553, 48631, 28935, 34948, 37004, 37684, 51094, 3695, 8183, 23085, 27843, 29158, 41543, 41735, 48638, 42371, 20451, 31757, 34809, 38320, 40738, 50711, 10559, 13527, 17605, 19800, 20498, 35177, 44178, 45351, 47278, 7828, 9345, 9496, 9944, 17247, 23398, 28682, 34564, 43578, 50665, 20565, 34591, 50862, 50918, 20, 4307, 10275, 15419, 19459, 23212, 44169, 48612, 50564, 28531, 43103, 50253, 9735, 16859, 27881, 31094, 38214, 51149, 9132, 28724, 33709, 36793, 40828, 48618, 48755, 646, 4387, 16578, 22801, 32336, 32618, 25669, 44795, 7453, 14460, 40840, 51628, 10921, 14492, 44077, 44145, 48958, 20290, 20361, 21445, 38457, 716, 22519, 23273, 34676, 39956, 49262, 2268, 6844, 10338, 15731, 29264, 41507, 43055, 48438, 34094, 25211, 47498, 48057, 21844, 22843, 26593, 31690, 33825, 43821, 6836, 7570, 9317, 34659, 49151, 51813, 7665, 8474, 15316, 20828, 26372, 30470, 33780, 40974, 41794, 48224, 5176, 15435, 25453, 35271, 37340, 44733, 48185, 48435, 50432, 38365, 28333, 44553, 14729, 15271, 17197, 26346, 48697, 42354, 44111, 16079, 23675, 25392, 26866, 35176, 39474, 32589, 42141, 1946, 12335, 27870, 3464, 15323, 23006, 25159, 32070, 40276, 10376, 16109, 31133, 50860, 5642, 25953, 50385, 25538, 26843, 27887, 32346, 40136, 14280, 25628, 27719, 40233, 531, 8222, 16326, 39511, 4108, 10045, 14693, 15156, 15278, 20120, 27909, 33840, 35721, 42964, 4957, 32670, 3815, 15825, 33182, 27902, 36540, 9547, 23001, 24604, 26319, 32610, 37165, 4474, 26449, 38332, 44790, 50861, 6496, 26460, 41376, 47910, 15272, 25907, 27764, 32494, 42891, 47666, 16279, 26708, 28616, 38268, 14491, 3441, 31651, 7085, 19767, 28243, 35040, 44301, 41498, 42484, 49401, 51552, 38147, 40851, 41134, 46912, 49755, 19755, 22691, 24872, 25670, 28455, 30610, 40116, 40703, 45190, 16865, 41971, 27588, 47249, 48154, 49039, 5171, 10044, 15863, 25832, 26385, 30296, 25116, 51739, 14463, 17200, 36595, 50487, 19501, 15221, 47738, 50001, 50447, 13589, 20973, 26458, 39806, 41855, 19457, 25591, 38495, 48833, 14701, 23233, 30997, 32011, 34018, 47827, 7560, 10030, 15144, 23000, 47015, 47644, 34584, 19920, 25299, 40926, 20106, 31828, 51556, 5581, 10021, 9723, 24661, 25892, 33847, 37104, 23545, 27648, 41310, 12138, 18923, 24672, 26498, 15732, 27720, 30965, 3001, 15945, 28245, 35172, 8130, 28888, 40412, 10167, 15439, 22301, 23381, 41670, 6261, 31326, 42916, 7334, 23563, 30478, 32650, 32731, 41190, 41893, 3940, 13490, 15136, 16078, 16322, 40400, 34120, 35424, 40657, 47045, 47891, 48473, 50404, 30646, 38634, 2217, 28590, 4953, 31729, 25638, 21970, 31240, 46655, 32310, 20959, 45598, 20814, 34728, 10031, 26650, 267, 19946, 28752, 42892, 15559, 41464, 8133, 22842, 33775, 47879, 22370, 11777, 24885, 37495, 50398, 51081, 16340, 23446, 27905, 23295, 35167, 44107, 39331, 40533, 6355, 19091, 22492, 28873, 9557, 32972, 40225, 13572, 1392, 22800, 51502, 17579, 20513, 32893, 22077, 50037, 23652, 34531, 9450, 39795, 40382, 40796, 43839, 4371, 9600, 41288, 39252, 41987, 17536, 40746, 46649, 8391, 22119, 40811, 47202, 14666, 40318, 47799, 38, 26352, 48127, 
    6824, 43194, 46679, 8147, 42094, 35444, 9546, 27367, 42772, 49437, 6823, 10450, 19573, 29313, 4083, 22200, 27866, 34771, 37233, 37977, 41547, 47382, 15587, 41446, 13392, 23637, 4082, 34954, 40543, 43548, 10465, 33476, 707, 13498, 45304, 46794, 16081, 17274, 41553, 26421, 43004, 32332, 3204, 21494, 34015, 31771, 15778, 34461, 15035, 15329, 6304, 15704, 33364, 13026, 15708, 15895, 20554, 25407, 25976, 6392, 15780, 1933, 42779, 45336, 28975, 40952, 45360, 51833, 15910, 14353, 24589, 35434, 40981, 13595, 33710, 15861, 39595, 23348, 34828, 35247, 51839, 64, 6232, 10000, 13571, 15405, 40288, 19461, 19523, 28241, 2676, 6303, 37929, 44987, 47864, 40291, 50255, 9187, 17573, 24869, 2272, 4272, 27198, 40266, 42091, 50919, 10413, 25519, 43549, 11251, 20378, 26903, 6718, 46851, 47251, 3105, 12734, 37720, 6863, 39662, 37485, 20208, 44084, 254, 19525, 51095, 51273, 7948, 48665, 40387, 50420, 10022, 28870, 29269, 12351, 25422, 7409, 49402, 32560, 15487, 20792, 31912, 33673, 23475, 27064, 37957, 41655, 44101, 26636, 15528, 34680, 39407, 4094, 38427, 40383, 10166, 42196, 33159, 333, 11908, 33065, 10173, 15860, 25100, 14288, 26347, 19472, 8055, 37816, 41241, 10229, 15318, 40572, 42998, 15286, 19768, 5097, 22630, 34385, 29050, 2273, 35445, 47494, 14368, 50838, 33678, 47670, 50842, 7849, 6624, 33896, 5671, 35586, 3038, 24636, 43353, 8101, 25929, 15123, 39529, 15824, 35438, 35174, 20385, 8580, 36092, 42199, 7924, 737, 5455, 34360, 48129, 11789, 25070, 21120, 41015, 45812, 294, 4488, 22385, 19769, 401, 24981, 35188, 15823, 39582, 42254, 1003, 31952, 25811, 41292, 51080, 7219, 10181, 26013, 24613, 24755, 45644, 28618, 41147, 9999, 49522, 44085, 31490, 25974, 29021, 32098, 32937, 15977, 47728, 15332, 35435, 51045, 51100, 7856, 24700, 36989, 343, 44088, 21299, 24195, 35785, 45620, 59, 31737, 48068, 13622, 50801, 13846, 23460, 15710, 15735, 27535, 34678, 7779, 8400, 40139, 12332, 49146, 6835, 35421, 40646, 7589, 20473, 21843, 24331, 31588, 35140, 50491, 420, 28268, 8491, 159, 461, 28572, 48095, 7970, 26287, 7935, 20074, 22007, 43206, 13371, 43500, 31327, 40688, 819, 37015, 25926, 51853, 28985, 25895, 7353, 40644, 6114, 8158, 41538, 41719, 23511, 51851, 3342, 25555, 38825, 140, 25527, 35123, 3099, 35019, 39098, 23183, 40648, 48103, 45608, 15709, 42048, 51108, 17269, 44086, 1017, 41839, 51848, 20441, 24686, 45346, 27533, 40929, 25864, 35411, 26282, 989, 31547, 40206, 7933, 17582, 8963, 50891, 22520, 8674, 25981, 26401, 47207, 14134, 15488, 49370, 4420, 40617, 7099, 21771, 35376, 41433, 39006, 30673, 6898, 3327, 24307, 39005, 42153, 24888, 33210, 22567, 32776, 24332, 33671, 15507, 8571, 30615, 35458, 51816, 20577, 8611, 51818, 10218, 51845, 39539, 5239, 17418, 12632, 15375, 49707, 29274, 28303, 24306, 37862, 51230, 51864, 17153, 38217, 20038, 26469, 6398, 10542, 13848, 38065, 51857, 10232, 2349, 7350, 15566, 42201, 42198, 35298, 8448, 12638, 48896, 27166, 22491, 7000, 34853, 10709, 22297, 21980, 39351, 24440, 41894, 25815, 47913, 35020, 25805, 8675, 8600, 13216, 24497, 7394, 3845, 51856, 15546, 8040, 24407, 13794, 32629, 24130, 45230, 51817, 51846, 3950, 51120, 8599, 6679, 15565, 32103, 45490, 39378, 42054, 51820, 23245, 38126, 22590, 29157, 24162, 39339, 51859, 47764, 13128, 42092, 7127, 35299, 42096, 25406, 41214, 33302, 43317, 8667, 28728, 39156, 8519, 10695, 42055, 7958, 8210, 42014, 38811, 31576, 40205, 7848, 28869, 25962, 10724, 40287, 42159, 32631, 42160, 21414, 22489, 25960, 38812, 47750, 31574, 25524, 8520, 47751, 24622, 20040, 23266, 40785, 21832, 25421, 8211, 40182, 40784, 8409, 40734, 22384, 23265, 21470, 38052, 7651, 7633, 42010, 40993, 38053, 38030, 38029, 42011, 38032, 25178, 38033, 40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
)

# How well does the acrobot baseline of only extrinsic reward do?
TspExperimentList["2-99_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity-no-combiner"] = TspExperimentList["2-98_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"].replace(  
NUM_TRIALS_PER_PROGRAM = 10,
REWARD_COMBINER_PROGRAMS_NAME = None,
FIXED_REWARD_COMBINER_PROGRAM_ID = None,      
)

# Lunarlander ICLR search!
TspExperimentList["2-100_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_lunarlander_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
    
ENVIRONMENT = "LunarLander-v2",
STEPS_PER_ROLLOUT = 10 * 1000,

NUM_TRIALS_PER_PROGRAM = 2,

SEARCH_PROGRAMS_EXPERIMENT_ID = "ss-1-knn-10-fv-1-pairs-early-termination-2-98",
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [18439, 25716, 26235, 32962, 35746, 35816, 37122, 42850, 43687, 43801, 50211, 50269, 50700, 51418, 51442, 2668, 16880, 19798, 25908, 31555, 34941, 40039, 43504, 51266, 9865, 10483, 10556, 13735, 16077, 16459, 16628, 20150, 27871, 39466, 43072, 3761, 7205, 8237, 11078, 11970, 14493, 15712, 16490, 17339, 19693, 19761, 24548, 25526, 27734, 30040, 30609, 31118, 34297, 35773, 36734, 42687, 47274, 48322, 50141, 2387, 6160, 6162, 14185, 16601, 19760, 20642, 20971, 25573, 26383, 34947, 37141, 45212, 49087, 49892, 14703, 15320, 16390, 16773, 22852, 22961, 23376, 24648, 28913, 32122, 37096, 37478, 44219, 49618, 50841, 50977, 656, 9790, 16509, 16710, 17002, 21740, 22458, 25247, 37538, 38548, 42915, 43511, 47685, 48454, 411, 4589, 16238, 17010, 19502, 19847, 25027, 28856, 32678, 33250, 34566, 37482, 39276, 40691, 41405, 42006, 43437, 43886, 19076, 20205, 20854, 38093, 40500, 40897, 43874, 45603, 49804, 50728, 9406, 9744, 11106, 14635, 15037, 19949, 28304, 28670, 30177, 33824, 34047, 34524, 35249, 37254, 38347, 39248, 43161, 43780, 51299, 8533, 19857, 23543, 28281, 31620, 32913, 34568, 34811, 35388, 38594, 40141, 42752, 48616, 51279, 12745, 19579, 19788, 37779, 47945, 1703, 14073, 28073, 35648, 35819, 38569, 50122, 50247, 51061, 9065, 16061, 16695, 26267, 31187, 31774, 32933, 35358, 40684, 43586, 44562, 47596, 49006, 7355, 16698, 20249, 20360, 23550, 25175, 28707, 29231, 31329, 32717, 40985, 42996, 43643, 47602, 50780, 81, 1045, 10907, 16587, 17918, 22462, 22467, 33895, 34007, 34104, 34559, 38606, 42132, 42953, 44081, 44248, 44673, 45483, 45638, 7159, 7438, 10335, 10701, 11094, 12492, 19334, 19845, 28992, 29222, 33498, 36096, 41521, 47263, 10194, 11320, 20334, 20968, 21948, 22502, 26098, 37558, 40476, 44302, 44793, 49655, 22054, 34562, 36254, 42489, 50353, 163, 6352, 9819, 15313, 16040, 23217, 26442, 28588, 28717, 28788, 37468, 38322, 38567, 39510, 51309, 2836, 2894, 3734, 8084, 9480, 16633, 25952, 26444, 26739, 39957, 40637, 43159, 44173, 49713, 2131, 6497, 11711, 14702, 20431, 21362, 25249, 28067, 29999, 31942, 32487, 34761, 37762, 38575, 48310, 50223, 51137, 10513, 11665, 14938, 17382, 18735, 19706, 24994, 28753, 30878, 31304, 38182, 42898, 49258, 51518, 9690, 14672, 16108, 31888, 50016, 20809, 29344, 29693, 34509, 35903, 36188, 36359, 42460, 46421, 19, 15315, 19656, 20277, 25230, 31348, 32379, 39643, 45181, 46647, 48791, 8376, 16921, 19183, 19480, 20993, 23685, 25039, 28758, 32195, 33768, 34001, 36729, 37454, 49817, 51569, 5951, 11038, 20228, 20817, 22887, 22958, 26334, 27812, 28653, 30975, 32533, 32689, 38343, 41745, 42940, 49860, 3168, 7457, 9551, 22642, 27632, 28639, 31852, 32125, 40401, 41570, 41898, 7582, 14430, 20594, 22076, 25177, 26195, 27888, 32515, 34849, 39611, 40973, 41374, 43180, 48253, 21688, 34788, 477, 10623, 17477, 20875, 27852, 37570, 44179, 47487, 11505, 12524, 20137, 31200, 32488, 39368, 44427, 46989, 49346, 1125, 16466, 22570, 26323, 35149, 36224, 37076, 37399, 38219, 38654, 41513, 42554, 42768, 49211, 50593, 50885, 51803, 129, 185, 5959, 12383, 19287, 22047, 26042, 27643, 29206, 37145, 38472, 40049, 40359, 41389, 41731, 41976, 45566, 310, 12489, 13783, 15666, 17304, 17576, 23051, 27158, 30810, 35133, 37024, 41268, 43881, 48732, 29901, 38386, 41629, 50196, 176, 2056, 15436, 23399, 25203, 28678, 33983, 34696, 37502, 39733, 41006, 41920, 942, 13017, 13582, 13611, 16295, 19878, 40780, 43374, 47525, 51229, 1943, 6564, 9618, 14641, 16342, 20706, 28075, 31241, 34540, 44007, 47396, 49457, 50563, 11164, 19978, 22289, 22317, 27854, 33204, 35306, 43425, 29016, 35329, 21708, 24288, 28433, 34547, 38550, 43042, 43757, 49307, 9941, 9943, 13337, 13479, 15702, 27066, 32302, 33669, 35450, 50673, 3951, 11933, 14465, 17375, 20475, 20860, 23518, 25158, 25289, 26162, 35264, 40069, 40745, 41478, 47901, 49294, 8273, 9687, 11202, 15293, 17102, 21396, 22838, 26526, 29242, 33940, 35976, 42832, 45614, 51689, 16298, 24521, 26510, 27717, 35173, 40126, 40988, 43053, 5052, 15951, 25375, 27875, 33680, 33718, 34532, 37445, 38453, 42403, 45357, 48418, 48619, 24030, 34622, 34665, 50699, 450, 1561, 15894, 20207, 20773, 23280, 23686, 29285, 32298, 34805, 38319, 39639, 16658, 23689, 31045, 32387, 33497, 49904, 10223, 11617, 15317, 28233, 33984, 36346, 37175, 38405, 38619, 41542, 45345, 49703, 50324, 462, 8090, 8637, 10379, 13345, 15295, 15663, 16654, 20974, 32291, 40301, 44339, 45540, 177, 992, 8665, 15084, 20313, 22447, 27763, 28743, 32078, 33111, 37161, 40707, 49880, 9554, 15234, 20332, 20856, 28272, 32009, 32491, 34442, 38413, 45509, 46680, 5561, 8516, 15947, 15973, 15978, 25463, 42331, 44180, 47369, 47529, 49184, 235, 4518, 9635, 12050, 24656, 25600, 28523, 28577, 31042, 36904, 38515, 41117, 46314, 47493, 48198, 50583, 924, 1467, 4381, 9131, 9275, 13499, 16812, 28325, 30652, 31001, 34453, 39309, 40243, 47284, 51110, 7258, 7934, 20090, 22552, 33173, 41652, 48599, 49260, 34053, 50052, 8895, 22430, 32990, 39641, 47935, 50513, 3425, 27531, 31393, 35178, 35447, 41037, 41046, 44212, 6362, 9356, 13529, 15451, 16711, 26529, 28649, 28918, 33668, 36306, 37106, 43235, 15319, 17065, 22270, 32474, 40127, 40579, 51103, 10337, 17021, 26512, 36948, 39581, 44114, 44837, 48211, 28452, 43110, 15946, 25144, 32393, 32534, 36616, 42183, 43867, 44729, 50915, 19455, 22988, 25481, 28316, 10860, 13623, 28410, 34783, 35666, 38028, 40138, 45607, 47331, 48520, 6214, 6625, 7620, 17300, 28255, 31069, 32790, 39302, 40786, 45193, 45337, 139, 5349, 26441, 27428, 33212, 38327, 43988, 50778, 28542, 34909, 37334, 42823, 50567, 19266, 37028, 41038, 41822, 43873, 45382, 10696, 17098, 17278, 17302, 22454, 23285, 27792, 35150, 37206, 43125, 43241, 51208, 51616, 14704, 23682, 32603, 32621, 34790, 35855, 36342, 41654, 44280, 4254, 8042, 31703, 40776, 40867, 43224, 15911, 15949, 37886, 39395, 40108, 46640, 49743, 49885, 50739, 6108, 28083, 28434, 28650, 36586, 6233, 10282, 14281, 17280, 29056, 33945, 38338, 44384, 48783, 51358, 10334, 21943, 27431, 41195, 3211, 12945, 22690, 26570, 31130, 31536, 32580, 34141, 37197, 39850, 43897, 47247, 8950, 15867, 19549, 22324, 22696, 50713, 6368, 21185, 22598, 33552, 39396, 39601, 43825, 10771, 15705, 15707, 16146, 37537, 38503, 17556, 29117, 33787, 33886, 35242, 22784, 37794, 8747, 9117, 11772, 40777, 40970, 46068, 50677, 12327, 27718, 31736, 38630, 40898, 2438, 10068, 27867, 34674, 36144, 38187, 49180, 50736, 1917, 11205, 18083, 20815, 50250, 50819, 27557, 34880, 35035, 36750, 38018, 39569, 49018, 50622, 7439, 9721, 26335, 28698, 32308, 40602, 41217, 41641, 47327, 48990, 13792, 19483, 19773, 22934, 34974, 35685, 46249, 6282, 6391, 15325, 25513, 37775, 41307, 43118, 33707, 44693, 49192, 15581, 7601, 16599, 26992, 27350, 14870, 27874, 49272, 8191, 15706, 17337, 23331, 33193, 34949, 49497, 51135, 5756, 15885, 24824, 36880, 20954, 23130, 24934, 47122, 7025, 28430, 43082, 44581, 16912, 24634, 38209, 42760, 45238, 4152, 14119, 20246, 28612, 31761, 32854, 50964, 995, 5843, 14692, 20196, 23211, 23622, 33862, 34798, 35041, 37016, 40553, 48631, 28935, 34948, 37004, 37684, 51094, 3695, 8183, 23085, 27843, 29158, 41543, 41735, 48638, 42371, 20451, 31757, 34809, 38320, 40738, 50711, 10559, 13527, 17605, 19800, 20498, 35177, 44178, 45351, 47278, 7828, 9345, 9496, 9944, 17247, 23398, 28682, 34564, 43578, 50665, 20565, 34591, 50862, 50918, 20, 4307, 10275, 15419, 19459, 23212, 44169, 48612, 50564, 28531, 43103, 50253, 9735, 16859, 27881, 31094, 38214, 51149, 9132, 28724, 33709, 36793, 40828, 48618, 48755, 646, 4387, 16578, 22801, 32336, 32618, 25669, 44795, 7453, 14460, 40840, 51628, 10921, 14492, 44077, 44145, 48958, 20290, 20361, 21445, 38457, 716, 22519, 23273, 34676, 39956, 49262, 2268, 6844, 10338, 15731, 29264, 41507, 43055, 48438, 34094, 25211, 47498, 48057, 21844, 22843, 26593, 31690, 33825, 43821, 6836, 7570, 9317, 34659, 49151, 51813, 7665, 8474, 15316, 20828, 26372, 30470, 33780, 40974, 41794, 48224, 5176, 15435, 25453, 35271, 37340, 44733, 48185, 48435, 50432, 38365, 28333, 44553, 14729, 15271, 17197, 26346, 48697, 42354, 44111, 16079, 23675, 25392, 26866, 35176, 39474, 32589, 42141, 1946, 12335, 27870, 3464, 15323, 23006, 25159, 32070, 40276, 10376, 16109, 31133, 50860, 5642, 25953, 50385, 25538, 26843, 27887, 32346, 40136, 14280, 25628, 27719, 40233, 531, 8222, 16326, 39511, 4108, 10045, 14693, 15156, 15278, 20120, 27909, 33840, 35721, 42964, 4957, 32670, 3815, 15825, 33182, 27902, 36540, 9547, 23001, 24604, 26319, 32610, 37165, 4474, 26449, 38332, 44790, 50861, 6496, 26460, 41376, 47910, 15272, 25907, 27764, 32494, 42891, 47666, 16279, 26708, 28616, 38268, 14491, 3441, 31651, 7085, 19767, 28243, 35040, 44301, 41498, 42484, 49401, 51552, 38147, 40851, 41134, 46912, 49755, 19755, 22691, 24872, 25670, 28455, 30610, 40116, 40703, 45190, 16865, 41971, 27588, 47249, 48154, 49039, 5171, 10044, 15863, 25832, 26385, 30296, 25116, 51739, 14463, 17200, 36595, 50487, 19501, 15221, 47738, 50001, 50447, 13589, 20973, 26458, 39806, 41855, 19457, 25591, 38495, 48833, 14701, 23233, 30997, 32011, 34018, 47827, 7560, 10030, 15144, 23000, 47015, 47644, 34584, 19920, 25299, 40926, 20106, 31828, 51556, 5581, 10021, 9723, 24661, 25892, 33847, 37104, 23545, 27648, 41310, 12138, 18923, 24672, 26498, 15732, 27720, 30965, 3001, 15945, 28245, 35172, 8130, 28888, 40412, 10167, 15439, 22301, 23381, 41670, 6261, 31326, 42916, 7334, 23563, 30478, 32650, 32731, 41190, 41893, 3940, 13490, 15136, 16078, 16322, 40400, 34120, 35424, 40657, 47045, 47891, 48473, 50404, 30646, 38634, 2217, 28590, 4953, 31729, 25638, 21970, 31240, 46655, 32310, 20959, 45598, 20814, 34728, 10031, 26650, 267, 19946, 28752, 42892, 15559, 41464, 8133, 22842, 33775, 47879, 22370, 11777, 24885, 37495, 50398, 51081, 16340, 23446, 27905, 23295, 35167, 44107, 39331, 40533, 6355, 19091, 22492, 28873, 9557, 32972, 40225, 13572, 1392, 22800, 51502, 17579, 20513, 32893, 22077, 50037, 23652, 34531, 9450, 39795, 40382, 40796, 43839, 4371, 9600, 41288, 39252, 41987, 17536, 40746, 46649, 8391, 22119, 40811, 47202, 14666, 40318, 47799, 38, 26352, 48127, 
    6824, 43194, 46679, 8147, 42094, 35444, 9546, 27367, 42772, 49437, 6823, 10450, 19573, 29313, 4083, 22200, 27866, 34771, 37233, 37977, 41547, 47382, 15587, 41446, 13392, 23637, 4082, 34954, 40543, 43548, 10465, 33476, 707, 13498, 45304, 46794, 16081, 17274, 41553, 26421, 43004, 32332, 3204, 21494, 34015, 31771, 15778, 34461, 15035, 15329, 6304, 15704, 33364, 13026, 15708, 15895, 20554, 25407, 25976, 6392, 15780, 1933, 42779, 45336, 28975, 40952, 45360, 51833, 15910, 14353, 24589, 35434, 40981, 13595, 33710, 15861, 39595, 23348, 34828, 35247, 51839, 64, 6232, 10000, 13571, 15405, 40288, 19461, 19523, 28241, 2676, 6303, 37929, 44987, 47864, 40291, 50255, 9187, 17573, 24869, 2272, 4272, 27198, 40266, 42091, 50919, 10413, 25519, 43549, 11251, 20378, 26903, 6718, 46851, 47251, 3105, 12734, 37720, 6863, 39662, 37485, 20208, 44084, 254, 19525, 51095, 51273, 7948, 48665, 40387, 50420, 10022, 28870, 29269, 12351, 25422, 7409, 49402, 32560, 15487, 20792, 31912, 33673, 23475, 27064, 37957, 41655, 44101, 26636, 15528, 34680, 39407, 4094, 38427, 40383, 10166, 42196, 33159, 333, 11908, 33065, 10173, 15860, 25100, 14288, 26347, 19472, 8055, 37816, 41241, 10229, 15318, 40572, 42998, 15286, 19768, 5097, 22630, 34385, 29050, 2273, 35445, 47494, 14368, 50838, 33678, 47670, 50842, 7849, 6624, 33896, 5671, 35586, 3038, 24636, 43353, 8101, 25929, 15123, 39529, 15824, 35438, 35174, 20385, 8580, 36092, 42199, 7924, 737, 5455, 34360, 48129, 11789, 25070, 21120, 41015, 45812, 294, 4488, 22385, 19769, 401, 24981, 35188, 15823, 39582, 42254, 1003, 31952, 25811, 41292, 51080, 7219, 10181, 26013, 24613, 24755, 45644, 28618, 41147, 9999, 49522, 44085, 31490, 25974, 29021, 32098, 32937, 15977, 47728, 15332, 35435, 51045, 51100, 7856, 24700, 36989, 343, 44088, 21299, 24195, 35785, 45620, 59, 31737, 48068, 13622, 50801, 13846, 23460, 15710, 15735, 27535, 34678, 7779, 8400, 40139, 12332, 49146, 6835, 35421, 40646, 7589, 20473, 21843, 24331, 31588, 35140, 50491, 420, 28268, 8491, 159, 461, 28572, 48095, 7970, 26287, 7935, 20074, 22007, 43206, 13371, 43500, 31327, 40688, 819, 37015, 25926, 51853, 28985, 25895, 7353, 40644, 6114, 8158, 41538, 41719, 23511, 51851, 3342, 25555, 38825, 140, 25527, 35123, 3099, 35019, 39098, 23183, 40648, 48103, 45608, 15709, 42048, 51108, 17269, 44086, 1017, 41839, 51848, 20441, 24686, 45346, 27533, 40929, 25864, 35411, 26282, 989, 31547, 40206, 7933, 17582, 8963, 50891, 22520, 8674, 25981, 26401, 47207, 14134, 15488, 49370, 4420, 40617, 7099, 21771, 35376, 41433, 39006, 30673, 6898, 3327, 24307, 39005, 42153, 24888, 33210, 22567, 32776, 24332, 33671, 15507, 8571, 30615, 35458, 51816, 20577, 8611, 51818, 10218, 51845, 39539, 5239, 17418, 12632, 15375, 49707, 29274, 28303, 24306, 37862, 51230, 51864, 17153, 38217, 20038, 26469, 6398, 10542, 13848, 38065, 51857, 10232, 2349, 7350, 15566, 42201, 42198, 35298, 8448, 12638, 48896, 27166, 22491, 7000, 34853, 10709, 22297, 21980, 39351, 24440, 41894, 25815, 47913, 35020, 25805, 8675, 8600, 13216, 24497, 7394, 3845, 51856, 15546, 8040, 24407, 13794, 32629, 24130, 45230, 51817, 51846, 3950, 51120, 8599, 6679, 15565, 32103, 45490, 39378, 42054, 51820, 23245, 38126, 22590, 29157, 24162, 39339, 51859, 47764, 13128, 42092, 7127, 35299, 42096, 25406, 41214, 33302, 43317, 8667, 28728, 39156, 8519, 10695, 42055, 7958, 8210, 42014, 38811, 31576, 40205, 7848, 28869, 25962, 10724, 40287, 42159, 32631, 42160, 21414, 22489, 25960, 38812, 47750, 31574, 25524, 8520, 47751, 24622, 20040, 23266, 40785, 21832, 25421, 8211, 40182, 40784, 8409, 40734, 22384, 23265, 21470, 38052, 7651, 7633, 42010, 40993, 38053, 38030, 38029, 42011, 38032, 25178, 38033, 40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
)

# How well does the lunar lander basline of only extrinsic reward do?
TspExperimentList["2-101_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity-no-combiner"] = TspExperimentList["2-100_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
REWARD_COMBINER_PROGRAMS_NAME = None,
FIXED_REWARD_COMBINER_PROGRAM_ID = None,      
)

# Acrobot search using programs of a variety of quality
TspExperimentList["2-101_acrobot_distribution_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-98_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [159, 11154, 10450, 43644, 47525, 42012, 38032, 24243, 50152, 32629, 24130, 34301, 42698, 7000, 42013, 42023, 42019, 14631, 39595, 28975, 47207, 25178, 31986, 40494, 17153, 41421, 36169, 49766, 47137, 3087, 26203, 28869, 33302, 25391, 3380, 51755, 51080, 36446, 42022, 41793, 43082, 19628, 40734, 42020, 43565, 51687, 42011, 51818, 26503, 7551, 8667, 20332, 23266, 25895, 29346, 9353, 7848, 23265, 40182, 44963, 40905, 4474, 42015, 28516, 14067, 47891, 29902, 7127, 35421, 2124, 21414, 40993, 35123, 47202, 46655, 45450, 7633, 30615, 21832, 42055, 38052, 21610, 42095, 33210, 26293, 51045, 25981, 14672, 15565, 44249, 5664, 42018, 25524, 2015, 42010, 8409, 42021, 31365, 333, 24306, 25421, 20040, 25070, 21626, 366, 35297, 20692, 22200, 39724, 1069, 1017, 15318, 15491, 36175, 6496, 32776, 27533, 24622, 22650, 31465, 14134, 28831, 24331, 38053, 47913, 39339, 929, 42017, 41794, 3321, 39378, 32631, 22384, 22590, 41806, 15332, 37106, 28427, 35298, 25805, 35921, 42196, 1492, 2349, 39109, 8158, 9983, 21038, 25892, 7584, 19472, 40785, 35020, 21394, 49370, 10695, 47381, 42159, 47424, 42092, 51594, 33029, 30004, 7958, 48068, 50608, 38029, 16283, 31669, 46439, 46592, 16832, 39896, 17929, 40691, 37699, 40688, 38031, 7651, 38405, 8519, 38030, 32937, 38552, 21470, 6679, 9894, 51108, 15546, 40070, 40784],
NUM_TRIALS_PER_PROGRAM = 5,
SPLIT_ACROSS_MACHINES = None,
)

# How well does our reward combiner compare on LL when it has fewer steps?
TspExperimentList["2-102_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity-no-combiner"] = TspExperimentList["2-100_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
REWARD_COMBINER_PROGRAMS_NAME = None,
FIXED_REWARD_COMBINER_PROGRAM_ID = None,      
)

# Ant on top 16 programs
TspExperimentList["2-103_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Ant-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
)

# Hopper on top 16 programs
TspExperimentList["2-103_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Hopper-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
)

# Ant on top 16 programs
TspExperimentList["2-104_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Ant-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
)

# Hopper on top 16 programs
TspExperimentList["2-104_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Hopper-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
)

# WITH DIFFERENT SEEDS:

# Ant on top 16 programs
TspExperimentList["2-105_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Ant-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 1,
)

# Hopper on top 16 programs
TspExperimentList["2-105_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Hopper-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 1,
)

# Ant on top 16 programs
TspExperimentList["2-106_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Ant-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 2,
)

# Hopper on top 16 programs
TspExperimentList["2-106_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Hopper-v2",
NUM_TRIALS_PER_PROGRAM = 1,
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [40070, 42017, 22645, 42013, 42020, 42012, 42023, 7551, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 2,
)

# FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS
# Ant on top 16 programs
TspExperimentList["2-107_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Ant-v2",
NUM_TRIALS_PER_PROGRAM = 1,
# NOTE THIS IS A DIFF LIST
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [42017, 42013, 42020, 42012, 42023, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 1,
FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS = True,
)

# Hopper on top 16 programs
TspExperimentList["2-107_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Hopper-v2",
NUM_TRIALS_PER_PROGRAM = 1,
# NOTE THIS IS A DIFF LIST
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [42017, 42013, 42020, 42012, 42023, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 1,
FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS = True,
)

# Ant on top 16 programs
TspExperimentList["2-108_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Ant-v2",
NUM_TRIALS_PER_PROGRAM = 1,
# NOTE THIS IS A DIFF LIST
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [42017, 42013, 42020, 42012, 42023, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 2,
FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS = True,
)

# Hopper on top 16 programs
TspExperimentList["2-108_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-97_mujoco_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
ENVIRONMENT = "Hopper-v2",
NUM_TRIALS_PER_PROGRAM = 1,
# NOTE THIS IS A DIFF LIST
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [42017, 42013, 42020, 42012, 42023, 38034, 42021, 42015, 42019, 38031, 42022, 42018, 42016],
SPLIT_ACROSS_MACHINES = None,
RANDOM_SEED_OFFSET = 2,
FIXED_CONTINUOUS_ACTION_PREDICTION_LOSS = True,
)
 


TspExperimentList["2-100_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"] = TspExperimentList["2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials"].replace(
    KNN_BUFFER_SIZE_SMALL = 100, 
    KNN_BUFFER_SIZE_LARGE = 1000,
    CURIOSITY_PROGRAMS_NAME = "intrinsic_reward_programs_v8",
)

# ICLR Rebuttal Experiments:

TspExperimentList["2-111_lunar-lander_diversity"] = \
    TspExperimentList["2-100_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
        SPLIT_ACROSS_MACHINES=False,
        RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [0, 1, 2, 5, 6, 7, 8, 13, 17, 21, 23, 30, 42, 46, 47, 49, 50, 51, 53, 54, 55, 63, 65, 69, 70, 89, 93, 102, 106, 107, 112, 117, 120, 125, 126, 168, 187, 189, 208, 220, 239, 255, 293, 309, 320, 323, 327, 332, 339, 352, 354, 364, 382, 383, 414, 422, 454, 468, 480, 482, 484, 491, 509, 511, 533, 594, 596, 601, 607, 616, 642, 681, 682, 715, 745, 759, 804, 814, 831, 894, 898, 930, 934, 1005, 1006, 1011, 1042, 1138, 1403, 1417, 1578, 1639, 1643, 1646, 1653, 1800, 1847, 1904, 1931, 1936, 1983, 1986, 2057, 2063, 2067, 2074, 2092, 2103, 2240, 2250, 2251, 2254, 2257, 2274, 2280, 2281, 2289, 2298, 2344, 2354, 2358, 2379, 2427, 2483, 2489, 2617, 2690, 2725, 2839, 2902, 3212, 3466, 3467, 3568, 3654, 4042, 4052, 4056, 4064, 4073, 4821, 5336, 5351, 6066, 6582, 6655, 6720, 6855, 7238, 7391, 7484, 7498, 7561, 7738, 8610, 8629, 8647, 8657, 8662, 8678, 8695, 8697, 8703, 8780, 8825, 8882, 8883, 8886, 8939, 9038, 9050, 9052, 9072, 9085, 9088, 9091, 9099, 9101, 9110, 9111, 9116, 9143, 9144, 9156, 9169, 9199, 9202, 9205, 9236, 9243, 9253, 9269, 9271, 9278, 9281, 9283, 9284, 9288, 9289, 9382, 9433, 9449, 9473, 9479, 9495, 9539, 9725, 9734, 9736, 9827, 9843, 10007, 10024, 10060, 10062, 10138, 10156, 10191, 10193, 10197, 10221, 10259, 10266, 10268, 10271, 10272, 10280, 10340, 10418, 10437, 10510, 10551, 10560, 10582, 10611, 10614, 10677, 10691, 10716, 10720, 10813, 10945, 11048, 11062, 11174, 11227, 11243, 11254, 11392, 11417, 11497, 11507, 11547, 11550, 11610, 11621, 11639, 11662, 11673, 11674, 11679, 11712, 11755, 11814, 11820, 11824, 11827, 11831, 11851, 11869, 11891, 11924, 11940, 11976, 12009, 12051, 12355, 13325, 13370, 13814, 13831, 13847, 13886, 13945, 13961, 13975, 14040, 14076, 14107, 14153, 14202, 14275, 14276, 14282, 14302, 14311, 14363, 14376, 14383, 14385, 14389, 14391, 14397, 14399, 14404, 14409, 14429, 14441, 14484, 14802, 14856, 14897, 14916, 14922, 14998, 15242, 15245, 15256, 15289, 15291, 15314, 15326, 15357, 15358, 15369, 15440, 15486, 15491, 15499, 15502, 15512, 15521, 15524, 15529, 15537, 15558, 15561, 15564, 15607, 15616, 15627, 15633, 15650, 15659, 15669, 15677, 15681, 15694, 15736, 15737, 15738, 15746, 15753, 15765, 15769, 15771, 15784, 15796, 15800, 15817, 15826, 15840, 15845, 15850, 15868, 15870, 15879, 15880, 15881, 15882, 15886, 15887, 15909, 15931, 15936, 15966, 15980, 15981, 16068, 16069, 16076, 16083, 16087, 16097, 16190, 16192, 16232, 16242, 16259, 16284, 16353, 16383, 16397, 16406, 16408, 16411, 16435, 16452, 16469, 16496, 16545, 16571, 16606, 16622, 16640, 16647, 16657, 16678, 16721, 16756, 16758, 16761, 16763, 16769, 16776, 16799, 16800, 16806, 16808, 16816, 16863, 16882, 16896, 16907, 16948, 16950, 16960, 16967, 16971, 16981, 17013, 17196, 17292, 17328, 17366, 17387, 17408, 17529, 17535, 17559, 17583, 17610, 17771, 17810, 17873, 18127, 18128, 18336, 18381, 18388, 18414, 18562, 18682, 18762, 18907, 18911, 18916, 18941, 18946, 19033, 19399, 19580, 19622, 19662, 19708, 19859, 19943, 20052, 20261, 20550, 20830, 21088, 21464, 21790, 21994, 22079, 22090, 22409, 22606, 22915, 23037, 23337, 23648, 23719, 23861, 23957, 24003, 24021, 24192, 24261, 24571, 25036, 25046, 25292, 25359, 25705, 25706, 25750, 26031, 26078, 26236, 26432, 26493, 26629, 26637, 26641, 26667, 26731, 26805, 26902, 27234, 27267, 27301, 27455, 27457, 27594, 27606, 27661, 27740, 27770, 27779, 28290, 28336, 28565, 28759, 28823, 28852, 28872, 28893, 29296, 29316, 29332, 29371, 29391, 29506, 29732, 29923, 29934, 29970, 29988, 30003, 30004, 30233, 30263, 30321, 30326, 30447, 30593, 30674, 30894, 31017, 31020, 31152, 31154, 31167, 31198, 31248, 31385, 31433, 31446, 31458, 31625, 31675, 31807, 32119, 32168, 32280, 32445, 32648, 32701, 32746, 32926, 33083, 33129, 33220, 33231, 33370, 33575, 33624, 33837, 33885, 33978, 34170, 34353, 34366, 34432, 34435, 34467, 34472, 34571, 34912, 34955, 34963, 35015, 35059, 35223, 35422, 35446, 35483, 35487, 35491, 35587, 35615, 35620, 35682, 35696, 35979, 36005, 36149, 36175, 36438, 36464, 36467, 36610, 36644, 36841, 36856, 36945, 37030, 37292, 37315, 37347, 37394, 37414, 37475, 37528, 37536, 37546, 37560, 37580, 37587, 37593, 37603, 37613, 37683, 37914, 37948, 37980, 38020, 38068, 38237, 38406, 38416, 38449, 38463, 38469, 38475, 38485, 38498, 38562, 38590, 38714, 39365, 39721, 40251, 40585, 40655, 40749, 40930, 40979, 41003, 41016, 41099, 41236, 41277, 41308, 41378, 41426, 41456, 41520, 41626, 41661, 41743, 41747, 41803, 41810, 41891, 42066, 42189, 42226, 42307, 42310, 42321, 42326, 42694, 42748, 42792, 42840, 43147, 43205, 43213, 43214, 43216, 43226, 43229, 43230, 43238, 43251, 43270, 43290, 43346, 43364, 43372, 43392, 43394, 43462, 43472, 43494, 43498, 43501, 43507, 43508, 43515, 43540, 43558, 43590, 43596, 43617, 43622, 43625, 43678, 43697, 43698, 43722, 43755, 43781, 43847, 43866, 43879, 43893, 43906, 43914, 43918, 43921, 43925, 43939, 43969, 44011, 44044, 44059, 44063, 44072, 44080, 44087, 44095, 44117, 44122, 44126, 44134, 44144, 44148, 44156, 44157, 44159, 44160, 44165, 44166, 44168, 44181, 44205, 44214, 44239, 44347, 44348, 44353, 44368, 44380, 44385, 44410, 44428, 44474, 44476, 44478, 44481, 44491, 44494, 44495, 44496, 44506, 44507, 44513, 44515, 44523, 44533, 44535, 44543, 44544, 44547, 44556, 44723, 44779, 44780, 44788, 44861, 44863, 44874, 44877, 44889, 44892, 44909, 44910, 44917, 44919, 44923, 44929, 44935, 44938, 44941, 44944, 44970, 44986, 44992, 44995, 44997, 45000, 45014, 45022, 45043, 45045, 45067, 45101, 45113, 45122, 45146, 45268, 45383, 45495, 45827, 45849, 45855, 46048, 46245, 46671, 46688, 46992, 46996, 47246, 47290, 47314, 47330, 47520, 47545, 47567, 47619, 47645, 47724, 47881, 47942, 47962, 47997, 48034, 48097, 48394, 48629, 48644, 48818, 48829, 48839, 48853, 49033, 49036, 49040, 49231, 49474, 49661, 49681, 49760, 49770, 49777, 49824, 49838, 50129, 50222, 50227, 50351, 50371, 50417, 50522, 50596, 50788, 50845, 50929, 50946, 50961, 50978, 50983, 50997, 51026, 51037, 51073, 51092, 51117, 51118, 51133, 51142, 51186, 51191, 51194, 51196, 51210, 51218, 51233, 51235, 51243, 51244, 51281, 51282, 51290, 51292, 51306, 51325, 51342, 51344, 51345, 51346, 51349, 51351, 51352, 51355, 51365, 51398, 51421, 51425, 51426, 51505, 51508, 51516, 51525, 51555, 51576, 51646, 51648, 51657, 51658, 51660, 51711, 51733, 51735, 51741, 51749, 51764, 51770, 51789, 51823, 51850, 51858, 51865]

    )

TspExperimentList["2-111_lunar-lander_diversity_combined"] = \
    TspExperimentList["2-111_lunar-lander_diversity"].replace(

    )

TspExperimentList["2-112_acrobot_diversity"] = \
    TspExperimentList["2-98_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
        SPLIT_ACROSS_MACHINES=False,
        RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [0, 1, 2, 5, 6, 7, 8, 13, 17, 21, 23, 30, 42, 46, 47, 49, 50, 51, 53, 54, 55, 63, 65, 69, 70, 89, 93, 102, 106, 107, 112, 117, 120, 125, 126, 168, 187, 189, 208, 220, 239, 255, 293, 309, 320, 323, 327, 332, 339, 352, 354, 364, 382, 383, 414, 422, 454, 468, 480, 482, 484, 491, 509, 511, 533, 594, 596, 601, 607, 616, 642, 681, 682, 715, 745, 759, 804, 814, 831, 894, 898, 930, 934, 1005, 1006, 1011, 1042, 1138, 1403, 1417, 1578, 1639, 1643, 1646, 1653, 1800, 1847, 1904, 1931, 1936, 1983, 1986, 2057, 2063, 2067, 2074, 2092, 2103, 2240, 2250, 2251, 2254, 2257, 2274, 2280, 2281, 2289, 2298, 2344, 2354, 2358, 2379, 2427, 2483, 2489, 2617, 2690, 2725, 2839, 2902, 3212, 3466, 3467, 3568, 3654, 4042, 4052, 4056, 4064, 4073, 4821, 5336, 5351, 6066, 6582, 6655, 6720, 6855, 7238, 7391, 7484, 7498, 7561, 7738, 8610, 8629, 8647, 8657, 8662, 8678, 8695, 8697, 8703, 8780, 8825, 8882, 8883, 8886, 8939, 9038, 9050, 9052, 9072, 9085, 9088, 9091, 9099, 9101, 9110, 9111, 9116, 9143, 9144, 9156, 9169, 9199, 9202, 9205, 9236, 9243, 9253, 9269, 9271, 9278, 9281, 9283, 9284, 9288, 9289, 9382, 9433, 9449, 9473, 9479, 9495, 9539, 9725, 9734, 9736, 9827, 9843, 10007, 10024, 10060, 10062, 10138, 10156, 10191, 10193, 10197, 10221, 10259, 10266, 10268, 10271, 10272, 10280, 10340, 10418, 10437, 10510, 10551, 10560, 10582, 10611, 10614, 10677, 10691, 10716, 10720, 10813, 10945, 11048, 11062, 11174, 11227, 11243, 11254, 11392, 11417, 11497, 11507, 11547, 11550, 11610, 11621, 11639, 11662, 11673, 11674, 11679, 11712, 11755, 11814, 11820, 11824, 11827, 11831, 11851, 11869, 11891, 11924, 11940, 11976, 12009, 12051, 12355, 13325, 13370, 13814, 13831, 13847, 13886, 13945, 13961, 13975, 14040, 14076, 14107, 14153, 14202, 14275, 14276, 14282, 14302, 14311, 14363, 14376, 14383, 14385, 14389, 14391, 14397, 14399, 14404, 14409, 14429, 14441, 14484, 14802, 14856, 14897, 14916, 14922, 14998, 15242, 15245, 15256, 15289, 15291, 15314, 15326, 15357, 15358, 15369, 15440, 15486, 15491, 15499, 15502, 15512, 15521, 15524, 15529, 15537, 15558, 15561, 15564, 15607, 15616, 15627, 15633, 15650, 15659, 15669, 15677, 15681, 15694, 15736, 15737, 15738, 15746, 15753, 15765, 15769, 15771, 15784, 15796, 15800, 15817, 15826, 15840, 15845, 15850, 15868, 15870, 15879, 15880, 15881, 15882, 15886, 15887, 15909, 15931, 15936, 15966, 15980, 15981, 16068, 16069, 16076, 16083, 16087, 16097, 16190, 16192, 16232, 16242, 16259, 16284, 16353, 16383, 16397, 16406, 16408, 16411, 16435, 16452, 16469, 16496, 16545, 16571, 16606, 16622, 16640, 16647, 16657, 16678, 16721, 16756, 16758, 16761, 16763, 16769, 16776, 16799, 16800, 16806, 16808, 16816, 16863, 16882, 16896, 16907, 16948, 16950, 16960, 16967, 16971, 16981, 17013, 17196, 17292, 17328, 17366, 17387, 17408, 17529, 17535, 17559, 17583, 17610, 17771, 17810, 17873, 18127, 18128, 18336, 18381, 18388, 18414, 18562, 18682, 18762, 18907, 18911, 18916, 18941, 18946, 19033, 19399, 19580, 19622, 19662, 19708, 19859, 19943, 20052, 20261, 20550, 20830, 21088, 21464, 21790, 21994, 22079, 22090, 22409, 22606, 22915, 23037, 23337, 23648, 23719, 23861, 23957, 24003, 24021, 24192, 24261, 24571, 25036, 25046, 25292, 25359, 25705, 25706, 25750, 26031, 26078, 26236, 26432, 26493, 26629, 26637, 26641, 26667, 26731, 26805, 26902, 27234, 27267, 27301, 27455, 27457, 27594, 27606, 27661, 27740, 27770, 27779, 28290, 28336, 28565, 28759, 28823, 28852, 28872, 28893, 29296, 29316, 29332, 29371, 29391, 29506, 29732, 29923, 29934, 29970, 29988, 30003, 30004, 30233, 30263, 30321, 30326, 30447, 30593, 30674, 30894, 31017, 31020, 31152, 31154, 31167, 31198, 31248, 31385, 31433, 31446, 31458, 31625, 31675, 31807, 32119, 32168, 32280, 32445, 32648, 32701, 32746, 32926, 33083, 33129, 33220, 33231, 33370, 33575, 33624, 33837, 33885, 33978, 34170, 34353, 34366, 34432, 34435, 34467, 34472, 34571, 34912, 34955, 34963, 35015, 35059, 35223, 35422, 35446, 35483, 35487, 35491, 35587, 35615, 35620, 35682, 35696, 35979, 36005, 36149, 36175, 36438, 36464, 36467, 36610, 36644, 36841, 36856, 36945, 37030, 37292, 37315, 37347, 37394, 37414, 37475, 37528, 37536, 37546, 37560, 37580, 37587, 37593, 37603, 37613, 37683, 37914, 37948, 37980, 38020, 38068, 38237, 38406, 38416, 38449, 38463, 38469, 38475, 38485, 38498, 38562, 38590, 38714, 39365, 39721, 40251, 40585, 40655, 40749, 40930, 40979, 41003, 41016, 41099, 41236, 41277, 41308, 41378, 41426, 41456, 41520, 41626, 41661, 41743, 41747, 41803, 41810, 41891, 42066, 42189, 42226, 42307, 42310, 42321, 42326, 42694, 42748, 42792, 42840, 43147, 43205, 43213, 43214, 43216, 43226, 43229, 43230, 43238, 43251, 43270, 43290, 43346, 43364, 43372, 43392, 43394, 43462, 43472, 43494, 43498, 43501, 43507, 43508, 43515, 43540, 43558, 43590, 43596, 43617, 43622, 43625, 43678, 43697, 43698, 43722, 43755, 43781, 43847, 43866, 43879, 43893, 43906, 43914, 43918, 43921, 43925, 43939, 43969, 44011, 44044, 44059, 44063, 44072, 44080, 44087, 44095, 44117, 44122, 44126, 44134, 44144, 44148, 44156, 44157, 44159, 44160, 44165, 44166, 44168, 44181, 44205, 44214, 44239, 44347, 44348, 44353, 44368, 44380, 44385, 44410, 44428, 44474, 44476, 44478, 44481, 44491, 44494, 44495, 44496, 44506, 44507, 44513, 44515, 44523, 44533, 44535, 44543, 44544, 44547, 44556, 44723, 44779, 44780, 44788, 44861, 44863, 44874, 44877, 44889, 44892, 44909, 44910, 44917, 44919, 44923, 44929, 44935, 44938, 44941, 44944, 44970, 44986, 44992, 44995, 44997, 45000, 45014, 45022, 45043, 45045, 45067, 45101, 45113, 45122, 45146, 45268, 45383, 45495, 45827, 45849, 45855, 46048, 46245, 46671, 46688, 46992, 46996, 47246, 47290, 47314, 47330, 47520, 47545, 47567, 47619, 47645, 47724, 47881, 47942, 47962, 47997, 48034, 48097, 48394, 48629, 48644, 48818, 48829, 48839, 48853, 49033, 49036, 49040, 49231, 49474, 49661, 49681, 49760, 49770, 49777, 49824, 49838, 50129, 50222, 50227, 50351, 50371, 50417, 50522, 50596, 50788, 50845, 50929, 50946, 50961, 50978, 50983, 50997, 51026, 51037, 51073, 51092, 51117, 51118, 51133, 51142, 51186, 51191, 51194, 51196, 51210, 51218, 51233, 51235, 51243, 51244, 51281, 51282, 51290, 51292, 51306, 51325, 51342, 51344, 51345, 51346, 51349, 51351, 51352, 51355, 51365, 51398, 51421, 51425, 51426, 51505, 51508, 51516, 51525, 51555, 51576, 51646, 51648, 51657, 51658, 51660, 51711, 51733, 51735, 51741, 51749, 51764, 51770, 51789, 51823, 51850, 51858, 51865]
  
    )

TspExperimentList["2-112_acrobot_diversity_combined"] = \
    TspExperimentList["2-112_acrobot_diversity"].replace(

    )

TspExperimentList["2-113_acrobot_diversity_combined"] = \
    TspExperimentList["2-112_acrobot_diversity"].replace(

    )


# WITH DIFFERENT SEEDS, Raw select best 16 meta-selected

# Ant on top 16 programs
TspExperimentList["2-114_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-105_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [12638, 33983, 20860, 35376, 35298, 12050, 43241, 38031, 42015, 43578, 44795, 7551, 42012, 16081, 50919, 42022]
)

# Hopper on top 16 programs
TspExperimentList["2-114_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-105_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [12638, 33983, 20860, 35376, 35298, 12050, 43241, 38031, 42015, 43578, 44795, 7551, 42012, 16081, 50919, 42022]
)

# Ant on top 16 programs
TspExperimentList["2-115_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-106_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [12638, 33983, 20860, 35376, 35298, 12050, 43241, 38031, 42015, 43578, 44795, 7551, 42012, 16081, 50919, 42022]
)

# Hopper on top 16 programs
TspExperimentList["2-115_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-106_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = [12638, 33983, 20860, 35376, 35298, 12050, 43241, 38031, 42015, 43578, 44795, 7551, 42012, 16081, 50919, 42022]
)


# WITH DIFFERENT SEEDS, Diversely select best 16

diverse_meta_selected_ids = [
    42012, 7551, 35176, 15709, 9132, 420, 14492, 11392, 
    43364, 15581, 50417, 8599, 15488, 16960, 51864, 15546]
# Ant on top 16 programs
TspExperimentList["2-116_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-105_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = diverse_meta_selected_ids
)

# Hopper on top 16 programs
TspExperimentList["2-116_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-105_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = diverse_meta_selected_ids
)

# Ant on top 16 programs
TspExperimentList["2-117_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-106_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = diverse_meta_selected_ids
)

# Hopper on top 16 programs
TspExperimentList["2-117_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"] = TspExperimentList["2-106_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"].replace(
RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE = diverse_meta_selected_ids
)

