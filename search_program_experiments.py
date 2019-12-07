
from mlca.helpers.experiment_params import ExperimentParameters, ExperimentParameterList
from typing import Optional
from dataclasses import dataclass

@dataclass
class SearchParams(ExperimentParameters):
    PROGRAMS_NAME: str
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID: str
    PREDICT_PERFORMANCE_EXPERIMENT_ID: Optional[str] = None

    # Simulating search
    NUM_SEARCHES: Optional[int] = None

    # Novelty experiments
    NOVELTY_BONUS: Optional[bool] = None
    NOVELTY_WEIGHT: Optional[int] = None
    NOVELTY_DISTANCE: Optional[str] = None

    # Batching
    PROGRAMS_PER_BATCH: Optional[int] = None
    BATCH_SELECTION: Optional[str] = None
    EPSILON_RANDOM_PROGRAM_SELECTION: Optional[float] = None

    # Early termination
    ENABLE_EARLY_TERMINATION: Optional[bool] = None
    EARLY_TERMINATION_CHECKING_FREQUENCY: Optional[int] = None
    NUM_STDEVS_DOWN: Optional[int] = None
    NUM_BEST_PROGRAMS: Optional[int] = None

    # Diversity
    DIVERSITY_FEATURE_PAIRS: Optional[int] = None
    DIVERSITY_PERF_THRESHOLD: Optional[int] = None
    DIVERSITY_DELTA: Optional[int] = None
    DIVERSITY_FEATURE_INPUT_OUTPUT: Optional[bool] = None


SearchExperimentList = ExperimentParameterList()

SearchExperimentList["ss"] = SearchParams(
    PROGRAMS_NAME = "programs_9",
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-44-15x15-ppo-5-rollouts-500-steps-gcloud-k80",
    NOVELTY_BONUS = False,
)

SearchExperimentList["ss-random"]=SearchExperimentList["ss"].replace(
    BATCH_SELECTION = "RANDOM",
    NUM_SEARCHES = 10,
    PROGRAMS_PER_BATCH = 1000,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs"]=SearchExperimentList["ss"].replace(
    BATCH_SELECTION = "SORT",
    EPSILON_RANDOM_PROGRAM_SELECTION = 0.1,
    PREDICT_PERFORMANCE_EXPERIMENT_ID = "knn-10-fv-1-pairs",
    NUM_SEARCHES = 10,
    PROGRAMS_PER_BATCH = 1000,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus-10"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = 10,
    NOVELTY_DISTANCE = "L2",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus-1"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = 1,
    NOVELTY_DISTANCE = "L2",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus--1"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = -1,
    NOVELTY_DISTANCE = "L2",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus--10"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = -10,
    NOVELTY_DISTANCE = "L2",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus-1-l1"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = 1,
    NOVELTY_DISTANCE = "L1",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus-1-l1normalized"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = 1,
    NOVELTY_DISTANCE = "L1Normalized",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-knn-bonus-10-l1normalized"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    NOVELTY_BONUS = True,
    NOVELTY_WEIGHT = 10,
    NOVELTY_DISTANCE = "L1Normalized",
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-60"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-60-15x15-ppo-5-rollouts-500-steps-lunar-lander", # 2-51-15x15-ppo-1-rollout-1000-steps-reward-combiner-new-program-list",
    ENABLE_EARLY_TERMINATION = True,
    EARLY_TERMINATION_CHECKING_FREQUENCY = 100,
    PROGRAMS_PER_BATCH = 100, # TODO: Increase to 1000 again, this is just because currently don't have many evaluations.
    NUM_SEARCHES = 1,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-62"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs"].replace(
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-62-15x15-ppo-5-rollouts-500-steps-lunar-lander",
    ENABLE_EARLY_TERMINATION = True,
    EARLY_TERMINATION_CHECKING_FREQUENCY = 12 * 8, # 12 because we do 12 ticks every thing
    # TODO: Increase to 1000 again, this is just because currently don't have many evaluations.
    PROGRAMS_PER_BATCH = 8 * 7 * 2, # 8 GPUs, 7 workers each, wait 2x that
    NUM_SEARCHES = 1,
    NUM_STDEVS_DOWN = 2,
    NUM_BEST_PROGRAMS = 60,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-80"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-62"].replace(
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials",
    NUM_STDEVS_DOWN = None, # 1.5,
    NUM_BEST_PROGRAMS = 2500,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-89"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-62"].replace(
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-89_15x15_new-ppo-real-batched-shared_2500-steps_5-trials-early-termination",
    NUM_STDEVS_DOWN = 2,  
    NUM_BEST_PROGRAMS = 625,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-96"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-62"].replace(
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
    
    NUM_STDEVS_DOWN = 2,
    NUM_BEST_PROGRAMS = 250,

    PROGRAMS_PER_BATCH = 8 * 7, 

    # 12 because we do 12 ticks every ppo update
    ENABLE_EARLY_TERMINATION = True,
    EARLY_TERMINATION_CHECKING_FREQUENCY = 12 * 9,

        DIVERSITY_FEATURE_PAIRS = 2,
        DIVERSITY_FEATURE_INPUT_OUTPUT = True,
)

SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-98"]=SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-96"].replace(
    TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-98_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity",

    NUM_STDEVS_DOWN = None,
    NUM_BEST_PROGRAMS = None,

    PROGRAMS_PER_BATCH = 8 * 7,
    PROGRAMS_NAME = 'programs_curiosity_7_v7',

    ENABLE_EARLY_TERMINATION = False,
    EARLY_TERMINATION_CHECKING_FREQUENCY = None,
)

SearchExperimentList["ss-knn_10-fv_1_pairs-early_termination-diversity-knn"] = \
    SearchExperimentList["ss-1-knn-10-fv-1-pairs-early-termination-2-96"].replace(
        TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
        NUM_SEARCHES = 1,

        BATCH_SELECTION = "DIVERSITY",
        DIVERSITY_FEATURE_PAIRS = 2,
        DIVERSITY_FEATURE_INPUT_OUTPUT = True,
        DIVERSITY_DELTA = 2.5,
        DIVERSITY_PERF_THRESHOLD = 400,
    )

SearchExperimentList["ss-knn_10-fv_1_pairs-early_termination-diversity-gp"] = \
    SearchExperimentList["ss-knn_10-fv_1_pairs-early_termination-diversity-knn"].replace(
        TEST_SYNTHESIZED_PROGRAMS_EXPERIMENT_ID = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
        PREDICT_PERFORMANCE_EXPERIMENT_ID = "gp_fv-1-pairs",
        NUM_SEARCHES = 1,
    )
