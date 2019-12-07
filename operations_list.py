"""
Specifices "operation list"s, which are lists of operations that can 
be composed to make programs. For example, we have separate operation lists
for reward combiners and intrinsic curiosity modules.
"""
from typing import List, Callable
from dataclasses import dataclass

from mlca.helpers.experiment_params import ExperimentParameters, ExperimentParameterList
import mlca.operations as operations
import mlca.program_types as program_types
from mlca.program import Program

# Intrinsic curiosity lists
# 8: Post refactor
# 7: Version for ICLR
# 6: Fix duplicate synthesis, remove FvRunningNorm
# 7: Add predict, fix some duplicate detection

# Reward combiner lists
# 5: Post refactor
# 4: Fix add data structures for running norm and variable buffer
# 3: Add IntrinsicExtrinsicWeightedNormalizedSum
# 2: Remove WeightedNormalizedSSum
# 1: Initial

"""TestSynthesizedProgramsExperimentParameters"""
@dataclass
class OperationsSet(ExperimentParameters):
    OPERATIONS: List[operations.Operation]
    REQUIRE_UPDATE_PROGRAM: bool   
    INITIAL_VARIABLES_FN: Callable
    MIN_PROGRAM_LENGTH: int
    MAX_PROGRAM_LENGTH: int

OperationsSetList = ExperimentParameterList()

def without(list, items_to_remove):
    items_to_remove = set(items_to_remove)
    return [l for l in list if l not in items_to_remove]


def initial_variables_for_curiosity():
    inputs = [
        operations.Variable(
            program_types.ImageTensor(), 0,
            "observation_image"),
        operations.Variable(
            program_types.FeatureVectorActionSpace(), 1,
            "action_one_hot"),
        operations.Variable(
            program_types.ImageTensor(), 2,
            "new_observation_image"),
    ]

    data_structures = [
        operations.Variable(
            program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32(), 0,
            "FCN_64_to_32", is_data_structure=True),
        operations.Variable(
            program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 1,
            "CNN1", is_data_structure=True),
        operations.Variable(
            program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 2,
            "CNN2", is_data_structure=True),
        operations.Variable(
            program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace(), 3,
            "FCN_32_to_Action_Space", is_data_structure=True),
        operations.Variable(
            program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(), 4,
            "FCN_Action_Space_to_32", is_data_structure=True),
        operations.Variable(
            program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(), 5,
            "FCN_64_to_Action_Space", is_data_structure=True),
        operations.Variable(
            program_types.EnsembleWeightsImageTo32(), 6,
            "EnsembleCNN1", is_data_structure=True),
        operations.Variable(
            program_types.EnsembleWeights32To32(), 7,
            "Ensemble_32_to_32_1", is_data_structure=True),
        operations.Variable(
            program_types.EnsembleWeights32AndActionTo32(), 8,
            "Ensemble_32_Action_to_32_1", is_data_structure=True),
        operations.Variable(
            program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVector32(), 13,
            "FCN_32_to_32", is_data_structure=True),
        operations.Variable(
            program_types.Policy(), 14,
            "Policy", is_data_structure=True),

        # can_only_use_once=True: 

        operations.Variable(
            program_types.RunningNormData(), 9,
            "RunningNormData", is_data_structure=True, can_only_use_once=True),
        operations.Variable(
            program_types.VariableBuffer(), 10,
            "VariableBuffer", is_data_structure=True, can_only_use_once=True),
        operations.Variable(
            program_types.FeatureVectorRunningNormData(), 11,
            "FeatureVectorRunningNormData", is_data_structure=True, can_only_use_once=True),
        operations.Variable(
            program_types.NearestNeighbor(), 12,
            "NearestNeighbor", is_data_structure=True, can_only_use_once=True),
    ]

    optimizers = [
        operations.Variable(program_types.AdamOptimizer(), 0,
            "Adam", is_optimizer=True)
    ]

    initial_program_list: List[Program] = []

    return inputs, data_structures, optimizers, initial_program_list


def initial_variables_for_reward_combiner():
    intrinsic_reward = operations.Variable(
        program_types.RealNumber(), 0,
        "intrinsic_reward", must_be_used=True)
    extrinsic_reward = operations.Variable(
        program_types.RealNumber(), 1,
        "extrinsic_reward", must_be_used=True)
    normalized_timestep = operations.Variable(
        program_types.RealNumber(), 2,
        "normalized_timestep")
    inputs = [
        intrinsic_reward,
        extrinsic_reward,
        normalized_timestep
    ]

    data_structures = [
        operations.Variable(
            program_types.RunningNormData(), 0,
            "RunningNormData", is_data_structure=True, can_only_use_once=True),
        operations.Variable(
            program_types.VariableBuffer(), 1,
            "VariableBuffer", is_data_structure=True, can_only_use_once=True),
    ]

    for i in range(0, 5):
        for p in [1, -1]:
            contant = p * (.5 ** i)
            data_structures.append(
                operations.Variable(
                    program_types.Constant(contant), len(data_structures),
                    f"constant_{contant}", is_constant=True, is_data_structure=True))

    optimizers: List[operations.Optimizer] = [

    ]

    initial_program_list = [
        Program([operations.Identity(intrinsic_reward)], [],
            inputs, data_structures, optimizers, 0
        ),
        Program([operations.Identity(extrinsic_reward)], [],
            inputs, data_structures, optimizers, 1
        ),
    ]

    return inputs, data_structures, optimizers, initial_program_list


OperationsSetList["reward_combiner_programs_v4"] = OperationsSet(
    REQUIRE_UPDATE_PROGRAM = False,
    INITIAL_VARIABLES_FN = initial_variables_for_reward_combiner,
    MIN_PROGRAM_LENGTH=2,
    MAX_PROGRAM_LENGTH=5,
    OPERATIONS = [
        operations.Add,
        operations.Max,
        operations.Min,
        operations.IntrinsicExtrinsicWeightedNormalizedSum,
        operations.RunningNorm,
        operations.RunningNormDontCenter,
        operations.VariableAsBuffer,
        operations.NormalDistribution,
        operations.Subtract,
        operations.Multiply,
        operations.SquareRootAbs,
        operations.RealNumberListMean,
    ]
)

OperationsSetList["programs_curiosity_7_v7"] = OperationsSet(
    # WARNING: This operations set might not be accurate; it's left here for backwards compatibility
    REQUIRE_UPDATE_PROGRAM = True,
    INITIAL_VARIABLES_FN = initial_variables_for_curiosity,
    MIN_PROGRAM_LENGTH=4,
    MAX_PROGRAM_LENGTH=7,
    OPERATIONS = [
        operations.Add,
        operations.RunningNorm,
        operations.RunningNormDontCenter,
        operations.VariableAsBuffer,
        operations.NearestNeighborRegressor,
        operations.SubtractOneTenth,
        operations.NormalDistribution,
        operations.Subtract,
        operations.SquareRootAbs,
        operations.FullyConnectedNetworkTwo32to32,
        operations.FullyConnectedNetworkTwo32toActionSpace,
        operations.FullyConnectedNetwork32toActionSpace,
        operations.FullyConnectedNetworkActionSpaceto32,
        operations.PredictFeatureVector32FromFeatureVector32,
        operations.CNN,
        operations.CNNDetachOutput,
        operations.CNNEnsemble,
        operations.FullyConnectedNetworkEnsemble32To32,
        operations.FullyConnectedNetworkEnsembleTwo32To32,
        operations.FullyConnectedNetworkEnsemble32AndActionTo32,
        operations.AddToLoss,
        operations.L2Norm,
        operations.L2Distance,
        operations.SoftmaxAndNLL,
        operations.DotProduct,
        operations.AddFeatureVector,
        operations.DetachFeatureVector,
        operations.RealNumberListMean,
        operations.ListVariance,
        operations.MeanOfFeatureVectorList,
        operations.FeatureVectorListL2Norm,
        operations.FeatureVectorListAverageL2DistanceToFeatureVector,
        operations.FeatureVectorListMinusFeatureVector,
        operations.IncrementCounter,
        operations.GetCounterValue,
    ]
)

OperationsSetList["intrinsic_reward_programs_v8"] = OperationsSet(
    REQUIRE_UPDATE_PROGRAM = True,
    INITIAL_VARIABLES_FN = initial_variables_for_curiosity,
    MIN_PROGRAM_LENGTH=4,
    MAX_PROGRAM_LENGTH=7,
    OPERATIONS = without(OperationsSetList["programs_curiosity_7_v7"].OPERATIONS, [
        operations.NearestNeighborRegressor, # Split into different types + fixed to be a proper regressor
        operations.VariableAsBuffer # Fixed to deal with batches
    ]) + [
        operations.NearestNeighborSmall,
        operations.NearestNeighborLarge,
        operations.NearestNeighborRegressorFixed,
        operations.ConditionalVAEReconstruction,
        operations.Policy,
        operations.VariableAsBufferCombined,
    ]
)

OperationsSetList["reward_combiner_programs_v5"] = OperationsSet(
    REQUIRE_UPDATE_PROGRAM = False,
    INITIAL_VARIABLES_FN = initial_variables_for_reward_combiner,
    MIN_PROGRAM_LENGTH=2,
    MAX_PROGRAM_LENGTH=4,
    OPERATIONS = OperationsSetList["reward_combiner_programs_v4"].OPERATIONS + [
        operations.RealNumberListLinearRegressionSlope,
        operations.Clip,
        operations.Square,
    ]
)

