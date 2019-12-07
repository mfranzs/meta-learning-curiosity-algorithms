from collections import OrderedDict
from mlca.helpers.experiment_params import ExperimentParameters, ExperimentParameterList
from enum import Enum
from typing import Optional, List, Any
from dataclasses import dataclass

@dataclass
class ProgramDistanceParams(ExperimentParameters):
    FEATURE_PAIRS: int
    TASK: str
    MODEL: str
    FEATURE_INPUT_OUTPUT: bool
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME: Optional[str]
    NEIGHBORS: Optional[int] = None

ProgramDistanceExperimentList = ExperimentParameterList()

ProgramDistanceExperimentList["mlp"] = ProgramDistanceParams(
    FEATURE_PAIRS = 1,
    NEIGHBORS = 10,
    TASK = "TEST_SINGLE",
    MODEL = "MLP",
    FEATURE_INPUT_OUTPUT = False,
    TEST_SYNTHESIZED_PROGRAMS_EXP_NAME="2-96_program-correlation-5",
)
