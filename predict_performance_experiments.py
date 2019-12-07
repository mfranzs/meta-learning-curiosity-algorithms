from collections import OrderedDict
from mlca.helpers.experiment_params import ExperimentParameters, ExperimentParameterList
from enum import Enum
from typing import Optional, List, Any
from dataclasses import dataclass

"""TestSynthesizedProgramsExperimentParameters"""
@dataclass
class PredictPerformanceParams(ExperimentParameters):
    FEATURE_PAIRS: int
    TASK: str
    MODEL: str
    FEATURE_INPUT_OUTPUT: bool
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME: Optional[str]
    NEIGHBORS: Optional[int] = None

PredictPerformanceExperimentList = ExperimentParameterList()

PredictPerformanceExperimentList["knn-10-fv-1-pairs"] = PredictPerformanceParams(
    FEATURE_PAIRS = 1,
    NEIGHBORS = 10,
    TASK = "TEST_SINGLE",
    MODEL = "KNN",
    FEATURE_INPUT_OUTPUT = False,
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME="2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
)

PredictPerformanceExperimentList["knn-10-fv-2-pairs"] = \
    PredictPerformanceExperimentList["knn-10-fv-1-pairs"].replace(
        FEATURE_PAIRS = 2,
    )


PredictPerformanceExperimentList["knn-10-fv-2-pairs-in-out"] = \
    PredictPerformanceExperimentList["knn-10-fv-2-pairs"].replace(
        FEATURE_INPUT_OUTPUT = 2,
    )


PredictPerformanceExperimentList["gp_fv-1-pairs"] = PredictPerformanceParams(
    FEATURE_PAIRS = 1,
    TASK = "TEST_SINGLE",
    MODEL = "GP",
    FEATURE_INPUT_OUTPUT = False,
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME="2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
)

PredictPerformanceExperimentList["gp_fv-2-pairs"] = PredictPerformanceParams(
    FEATURE_PAIRS = 2,
    TASK = "TEST_SINGLE",
    MODEL = "GP",
    FEATURE_INPUT_OUTPUT = False,
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME="2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
)

PredictPerformanceExperimentList["gp_fv-2-pairs-in-out"] = PredictPerformanceParams(
    FEATURE_PAIRS = 2,
    TASK = "TEST_SINGLE",
    MODEL = "GP",
    FEATURE_INPUT_OUTPUT = True,
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME="2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity",
)


# experiments = OrderedDict([
#     ("programs-9", {
#       __ABSTRACT__ = True,
#       PROGRAMS_NAME = "programs_9",
#       DATA_EXP_NAME = "" # programs_9_evaluations.py
#     }),

#     ("knn-100-fv-3-pairs", {
#       __PARENT__ = "programs-9",
#       TASK = "TEST_SINGLE",
#       MODEL = "KNN",
#       FEATURE_PAIRS = 3,
#       FEATURE_INPUT_OUTPUT = False,
#       NEIGHBORS = 100,
#     }),
#     ("knn-100-fv-2-pairs", {
#         __PARENT__ = "knn-100-fv-3-pairs",
#         FEATURE_PAIRS = 2,
#     }),
#     ("knn-100-fv-1-pairs", {
#         __PARENT__ = "knn-100-fv-3-pairs",
#         FEATURE_PAIRS = 1,
#     }),

#     ("knn-10-fv-3-pairs", {
#         __PARENT__ = "knn-100-fv-3-pairs",
#         FEATURE_PAIRS = 3,
#         NEIGHBORS = 10,
#     }),
#     ("knn-10-fv-2-pairs", {
#         __PARENT__ = "knn-10-fv-3-pairs",
#         FEATURE_PAIRS = 2,
#     }),
#     ("knn-10-fv-1-pairs", {
#         __PARENT__ = "knn-10-fv-3-pairs",
#         FEATURE_PAIRS = 1,
#     }),

#     ("rf-100-fv-3-pairs", {
#         __PARENT__ = "programs-9",
#         TASK = "TEST_SINGLE",
#         MODEL = "RandomForest",
#         NUM_ESTIMATORS = 100,
#         FEATURE_PAIRS = 3,
#         FEATURE_INPUT_OUTPUT = False,
#     }),
#     ("rf-100-fv-2-pairs", {
#         __PARENT__ = "rf-100-fv-3-pairs",
#         FEATURE_PAIRS = 2,
#     }),
#     ("rf-100-fv-1-pairs", {
#         __PARENT__ = "rf-100-fv-3-pairs",
#         FEATURE_PAIRS = 1,
#     }),

#     ("linear-3-pairs", {
#         __PARENT__ = "programs-9",
#         TASK = "TEST_SINGLE",
#         MODEL = "Linear",
#         FEATURE_PAIRS = 3,
#         FEATURE_INPUT_OUTPUT = False,
#     }),
#     ("linear-2-pairs", {
#         __PARENT__ = "linear-3-pairs",
#         FEATURE_PAIRS = 2,
#     }),
#     ("linear-1-pairs", {
#         __PARENT__ = "linear-3-pairs",
#         FEATURE_PAIRS = 1,
#     }),

#     ("linear-1-pairs-sweep", {
#         __PARENT__ = "linear-1-pairs",
#         TASK = "TRAINING_POINTS_PLOT"
#     }),

#     ("linear-2-pairs-sweep", {
#         __PARENT__ = "linear-2-pairs",
#         TASK = "TRAINING_POINTS_PLOT"
#     }),

#     ("knn-10-fv-1-pairs-sweep", {
#         __PARENT__ = "knn-10-fv-1-pairs",
#         TASK = "TRAINING_POINTS_PLOT"
#     }),

#     # -----

#     ("knn-10-fv-1-pairs-input-outputs", {
#         __PARENT__ = "knn-10-fv-1-pairs",
#         FEATURE_INPUT_OUTPUT = True,
#     }),

#     ("linear-2-pairs-input-outputs", {
#         __PARENT__ = "linear-2-pairs",
#         FEATURE_INPUT_OUTPUT = True,
#     }),

#     ("knn-5-fv-1-pairs", {
#         __PARENT__ = "knn-10-fv-1-pairs",
#         NEIGHBORS = 5,
#     }),

#     ("knn-15-fv-1-pairs", {
#         __PARENT__ = "knn-10-fv-1-pairs",
#         NEIGHBORS = 15,
#     }),
# ])
