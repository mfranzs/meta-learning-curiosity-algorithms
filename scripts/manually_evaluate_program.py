"""
Helper code that you can use to manually evaluate a set of programs
without running a full search.
"""

import pickle
import multiprocessing
import json
import pprint

import torch.multiprocessing as mp
import torch 

import mlca.helpers.config
import mlca.helpers.debug
from mlca.test_synthesized_programs_experiments import TspParams, TspExperimentList
from mlca.run_agent import evaluate_program_in_environment
from mlca.scripts.analyze_synthesized_programs import _stats_for_program, load_curiosity_program_data
# from mlca.scripts.re_evaluate_program import evaluate_program_on_multiple_rollouts
from mlca.operations import *
import mlca.program_types as program_types
from mlca.program import Program


def main():
    parser = mlca.helpers.config.argparser()
    parser.add_argument("--trials", type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--workers_per_gpu', default=1, type=int)
    parser.add_argument('--print_intermediate', default=False, type=bool)
    parser.add_argument('--random_seed_offset', default=None, type=int)
    args = parser.parse_args()

    params = TspExperimentList[args.experiment_id]

    with params:

        reward_combiner_program, reward_combiner_program_inputs, \
            reward_combiner_data_structures, reward_combiner_optimizers = get_reward_combiner_program(
                params)

        if args.trials is not None:
            TspParams.current().NUM_TRIALS_PER_PROGRAM = args.trials
            print("WARNING: OVERRIDING NUM_TRIALS_PER_PROGRAM")

        if args.random_seed_offset is not None:
            TspParams.current().RANDOM_SEED_OFFSET = args.random_seed_offset
            print("WARNING: OVERRIDING RANDOM_SEED_OFFSET")

        batch = {
            # "build_2_96_0": build_2_96_0,
            # "build_2_96_1": build_2_96_1,

            "noise": build_random_noise_program,
            "one": build_constant_one_program,
            "zero": build_constant_zero_program,
            "negative_one": build_constant_negative_one_program,

            # # "2_84_0": build_2_84_0,
            # # "2_84_1": build_2_84_1,
            # "2_84_2": build_2_84_2,
            # "2_84_3": build_2_84_3,
            # # "2_84_4": build_2_84_4,
            # # "2_84_5": build_2_84_5,
            # # "2_84_6": build_2_84_6,
            # # "2_84_7": build_2_84_7,
            # # "2_84_8": build_2_84_8,
            # # "2_84_9": build_2_84_9,

            "disagreement": build_disagreement_program,
            "inverse": build_inverse_program,
            "random_distillation": build_random_network_distillation_program,
            "best": build_best_program,
        }

        pool = multiprocessing.Pool(args.num_gpus * args.workers_per_gpu)

        worker_args = []
        for selected_index, program_name in enumerate(batch):
            print(program_name)

            program, program_inputs, data_structures, optimizers, = batch[program_name]()

            worker_args.append((
                program, 
                None,
                None,
                selected_index,
                params,
                None,
                ( args.num_gpus, reward_combiner_program, None),
                None))

        print("Starting pool")
            
        results_data = [
            evaluate_program_in_environment(*w) for w in worker_args
        ]
        # results_data = pool.map(
        #     evaluate_program_in_environment, worker_args)

        pool.close()
        pool.terminate()

        for program_name, result_data in zip(batch, results_data):
            print(program_name)
            print(result_data.stats)

        with open(f"pickles/manually_evaluate_program_{args.experiment_id}.txt", 'w') as output:
            pp = pprint.PrettyPrinter(indent=4)
            for program_name, result_data in zip(batch, results_data):
                output.write(program_name)
                output.write("\n")
                output.write("\t" + str(result_data.stats)) 
                output.write("\n")
            output.write(pp.pformat(params))
            output.write(pp.pformat([r for r in results_data]))

def get_reward_combiner_program(params):

    if TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID is None:
        print("No reward combiner")
        reward_combiner_program, reward_combiner_program_inputs, \
            reward_combiner_data_structures, reward_combiner_optimizers = None, None, None, None
    else:
        pre_evaluated_data, _, \
            curiosity_programs, curiosity_program_inputs, \
            curiosity_data_structures, curiosity_optimizers, \
            reward_combiner_programs, reward_combiner_program_inputs, \
            reward_combiner_data_structures, reward_combiner_optimizers, _ = load_curiosity_program_data(
                TspParams.current().CURIOSITY_PROGRAMS_NAME,
                TspParams.current().REWARD_COMBINER_PROGRAMS_NAME,
                TspParams.current()._experiment_id,
                TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID)
        reward_combiner_program = reward_combiner_programs[
            TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID] 
        print("Fixed reward combiner")
        print(reward_combiner_program)

    # intrinsic_reward = Variable(
    #             program_types.RealNumber(), 0,
    #             "intrinsic_reward", must_be_used=True)
    # extrinsic_reward = Variable(
    #                 program_types.RealNumber(), 1,
    #                 "extrinsic_reward", must_be_used=True)
    # normalized_timestep = Variable(
    #                     program_types.RealNumber(), 2,
    #                     "normalized_timestep")
    # constant_1 = Variable(
    #                         program_types.Constant(1), 0,
    #                         f"constant_1", is_constant=True, is_data_structure=True)
    # reward_combiner_program_inputs = [
    #                         intrinsic_reward,
    #                         extrinsic_reward,
    #                         normalized_timestep
    #                     ]
    # reward_combiner_data_structures = [constant_1]
    # reward_combiner_optimizers = []
    # # TODO: Clean up this manual mess

    # #    if params["FIXED_REWARD_COMBINER_PROGRAM_ID"] else None
    # a = Add(constant_1, intrinsic_reward)
    # b = Subtract(a, normalized_timestep)
    # c = IntrinsicExtrinsicWeightedNormalizedSum(b, intrinsic_reward, normalized_timestep, extrinsic_reward)
    # # update only:
    # reward_combiner_program = Program([a, b, c], [])
    return reward_combiner_program, reward_combiner_program_inputs, \
        reward_combiner_data_structures, reward_combiner_optimizers

def build_FAST_program():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()

    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity
    # Index 42016
    # {'mean_performance': 610.4, 'mean_performance_std': 8.234561311933989, 'trial_mean(rollout_mean(max_episode_in_rollout))_performance': 610.4, 'trial_mean(rollout_mean(max_episode_in_rollout))_performance_std': 8.234561311933989, 'trial_mean(rollout_mean(last_episode_in_rollout))_performance': 610.4, 'trial_mean(rollout_mean(last_episode_in_rollout))_performance_std': 8.234561311933989}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {CNN1, new_observation_image}
    b = CNN(new_observation_image, CNN1)
    # NonNegativeNumber           {CNN1, new_observation_image, observation_image}
    c = L2Distance(b, a)
    # update only:
    # NonNegativeNumber           {CNN1, new_observation_image, action_one_hot}
    d = SoftmaxAndNLL(b, action_one_hot)
    # Void                        {CNN1, new_observation_image, action_one_hot}
    e = AddToLoss(d)
    program = Program([a, b, c], [d, e],
        program_inputs, data_structures, optimizers, -1, "FAST"
    )
    # -----------------

    return program, program_inputs, data_structures, optimizers


def build_double_predict_program():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()

    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity
    # Index 22645
    # {'mean_performance': 573.6, 'mean_performance_std': 12.92625235712192, 'trial_mean(rollout_mean(max_episode_in_rollout))_performance': 573.6, 'trial_mean(rollout_mean(max_episode_in_rollout))_performance_std': 12.92625235712192, 'trial_mean(rollout_mean(last_episode_in_rollout))_performance': 573.6, 'trial_mean(rollout_mean(last_episode_in_rollout))_performance_std': 12.92625235712192}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {CNN2, observation_image}
    b = CNN(observation_image, CNN2)
    # FeatureVector32             {new_observation_image, CNN2}
    c = CNN(new_observation_image, CNN2)
    # FeatureVector32             {FCN_32_to_32, CNN2, observation_image, CNN1}
    d = PredictFeatureVector32FromFeatureVector32(b, a, FCN_32_to_32)
    # FeatureVector32             {FCN_32_to_32, observation_image, new_observation_image, CNN2}
    e = PredictFeatureVector32FromFeatureVector32(c, b, FCN_32_to_32)
    # NonNegativeNumber           {CNN1, FCN_32_to_32, new_observation_image, CNN2, observation_image}
    f = L2Distance(e, d)
    # update only:
    program = Program([a, b, c, d, e, f], [],
        program_inputs, data_structures, optimizers, -1, "double_predict"
    )
    # -----------------
    
    return program, program_inputs, data_structures, optimizers


def build_best_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image",
        short_name="s(t)")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot",
        short_name="a(t)")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image",
        short_name="s(t+1)")

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    FCN_64_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32(), 3, "FCN_64_to_32", True)
    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True,
        short_name="θ{2}: 𝕊 → 𝔽")
    FCN_32_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace(), 5, "FCN_32_to_ACTION_SPACE", True)
    FCN_Action_Space_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(), 6, "FCN_Action_Space_to_32", True)
    FCN_64_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(), 7, "FCN_64_to_ACTION_SPACE", True)
    ENSEMBLE_CNN1 = Variable(
        program_types.EnsembleWeightsImageTo32(), 8, "ENSEMBLE_CNN1", True,
        short_name="θ{1}: 𝕊 → [𝔽]")
    EnsembleFCN1 = Variable(
        program_types.EnsembleWeights32To32(), 9, "ENSEMBLE_CNN1", True)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeights32AndActionTo32(), 9)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeightsTwo32To32(), 9)
    data_structures = [FCN_64_to_32, CNN1, FCN_32_to_ACTION_SPACE, FCN_Action_Space_to_32,
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1]

    Adam = Variable(program_types.AdamOptimizer(), 10, "Adam")
    optimizers = [Adam]


    # -----------------
    # Experiment: 2-28-15x15-ppo-5-rollouts-500-steps--version_b_small
    # Program 14801
    # {'mean': 135.18, 'std': 35.95958286743605}
    # FeatureVector32             {new_observation_image, CNN1}
    a = CNN(new_observation_image, CNN1)
    # ListFeatureVector32         {new_observation_image, ENSEMBLE_CNN1}
    b = CNNEnsemble(new_observation_image, ENSEMBLE_CNN1)
    # NonNegativeNumber           {new_observation_image, CNN1, ENSEMBLE_CNN1}
    c = FeatureVectorListAverageL2DistanceToFeatureVector(b, a)
    # update only:
    # FeatureVector32             {new_observation_image, CNN1}
    # d_1 = CNN(new_observation_image, CNN1)
    d = DetachFeatureVector(a)
    # ListFeatureVector32         {new_observation_image, CNN1, ENSEMBLE_CNN1}
    e = FeatureVectorListMinusFeatureVector(b, d)
    # ListRealNumber              {new_observation_image, CNN1, ENSEMBLE_CNN1}
    f = FeatureVectorListL2Norm(e)
    # RealNumber                  {new_observation_image, CNN1, ENSEMBLE_CNN1}
    g = RealNumberListSum(f)
    # NonNegativeNumber           {new_observation_image, CNN1, ENSEMBLE_CNN1}
    h = SquareRootAbs(g)
    # Void                        {new_observation_image, CNN1, ENSEMBLE_CNN1, Adam}
    i = AddToLoss(h)
    # program = Program([a, b, c], [d, e, f, g, h, i])

    a = CNN(new_observation_image, CNN1)                         # FeatureVector32             {CNN1, new_observation_image}
    b = CNNEnsemble(new_observation_image, ENSEMBLE_CNN1)         # ListFeatureVector32         {EnsembleCNN1, new_observation_image}
    c = FeatureVectorListAverageL2DistanceToFeatureVector(b, a)  # NonNegativeNumber           {CNN1, EnsembleCNN1, new_observation_image}
    # update only: 
    d = CNNDetachOutput(new_observation_image, CNN1)             # FeatureVector32             {CNN1, new_observation_image}
    e = FeatureVectorListMinusFeatureVector(b, d)                # ListFeatureVector32         {CNN1, EnsembleCNN1, new_observation_image}
    f = FeatureVectorListL2Norm(e)                               # ListRealNumber              {CNN1, EnsembleCNN1, new_observation_image}
    g = RealNumberListSum(f)                                     # RealNumber                  {CNN1, EnsembleCNN1, new_observation_image}
    h = SquareRootAbs(g)                                         # NonNegativeNumber           {CNN1, EnsembleCNN1, new_observation_image}
    i = AddToLoss(h)                                   # Void                        {Adam, CNN1, EnsembleCNN1, new_observation_image}
    program = Program([a, b, c], [d, e, f, g, h, i],
        program_inputs, data_structures, optimizers, -1, "best"
    )

    # -----------------

    return program, program_inputs, data_structures, optimizers

def get_2_84_datastructures():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image",
        short_name="s(t)")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot",
        short_name="a(t)")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image",
        short_name="s(t+1)")

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    FCN_64_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32(), 3, "FCN_64_to_32", True)
    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True,
        short_name="θ{1}: 𝕊 → 𝔽")
    FCN_32_to_Action_Space = Variable(
        program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace(), 5, "FCN_32_to_ACTION_SPACE", True)
    FCN_Action_Space_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(), 6, "FCN_Action_Space_to_32", True)
    FCN_64_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(), 7, "FCN_64_to_ACTION_SPACE", True)
    ENSEMBLE_CNN1 = Variable(
        program_types.EnsembleWeightsImageTo32(), 8, "ENSEMBLE_CNN1", True)
    EnsembleFCN1 = Variable(
        program_types.EnsembleWeights32To32(), 9, "ENSEMBLE_CNN1", True)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeights32AndActionTo32(), 9)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeightsTwo32To32(), 9)

    FCN_32_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVector32(), 11, "FCN_32_to_32", True,
        short_name="θ{2}: 𝔽 → 𝔽")
    RunningNormData = Variable(
        program_types.RunningNormData(), 12,
        "RunningNormData", is_data_structure=True, can_only_use_once=True)
    VariableBuffer = Variable(
        program_types.VariableBuffer(), 13,
        "VariableBuffer", is_data_structure=True, can_only_use_once=True)
    FeatureVectorRunningNormData = Variable(
        program_types.FeatureVectorRunningNormData(), 14,
        "FeatureVectorRunningNormData", is_data_structure=True, can_only_use_once=True)
    NearestNeighbor = Variable(
        program_types.NearestNeighbor(), 15,
        "NearestNeighbor", is_data_structure=True, can_only_use_once=True)
    CNN2 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 16, "CNN2", True,
        short_name="θ{3}: 𝕊 → 𝔽")
    Ensemble_32_Action_to_32_1 = Variable(
            program_types.EnsembleWeights32AndActionTo32(), 17,
            "Ensemble_32_Action_to_32_1", is_data_structure=True)

    data_structures = [FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32,
                    FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32,
                       RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1]

    Adam = Variable(program_types.AdamOptimizer(), 10, "Adam")
    optimizers = [Adam]

    return program_inputs, data_structures, optimizers

def build_2_96_0():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()

    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity
    # Index 7848
    # {'mean_performance': 463.4, 'mean_performance_std': 54.90471746580616}
    # ------
    a = FullyConnectedNetworkActionSpaceto32(action_one_hot, FCN_Action_Space_to_32) # FeatureVector32             {FCN_Action_Space_to_32, action_one_hot}
    b = CNN(observation_image, CNN1)                             # FeatureVector32             {CNN1, observation_image}
    c = CNN(new_observation_image, CNN1)                         # FeatureVector32             {new_observation_image, CNN1}
    d = PredictFeatureVector32FromFeatureVector32(a, b, FCN_32_to_32) # FeatureVector32             {CNN1, FCN_32_to_32, FCN_Action_Space_to_32, observation_image, action_one_hot}
    e = NearestNeighborRegressor(d, b, NearestNeighbor)          # FeatureVector32             {CNN1, NearestNeighbor, FCN_32_to_32, FCN_Action_Space_to_32, observation_image, action_one_hot}
    f = L2Distance(c, e)                                         # NonNegativeNumber           {new_observation_image, NearestNeighbor, CNN1, FCN_32_to_32, FCN_Action_Space_to_32, observation_image, action_one_hot}
    # update only: 
    program = Program([a, b, c, d, e, f], [], 
        program_inputs, data_structures, optimizers, -1, "2_96_0"
    )
    # -----------------

    return program, program_inputs, data_structures, optimizers

def build_2_96_1():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()

    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity
    # Index 35785
    # {'mean_performance': 337.8, 'mean_performance_std': 71.05344467371023}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN1}
    b = CNN(new_observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN1, NearestNeighbor, observation_image}
    c = NearestNeighborRegressor(b, a, NearestNeighbor)
    # FeatureVector32             {new_observation_image, CNN1, FCN_32_to_32, observation_image}
    d = PredictFeatureVector32FromFeatureVector32(b, a, FCN_32_to_32)
    # NonNegativeNumber           {NearestNeighbor, CNN1, FCN_32_to_32, observation_image, new_observation_image}
    e = L2Distance(d, c)
    # update only:
    # NonNegativeNumber           {action_one_hot, CNN1, observation_image}
    f = SoftmaxAndNLL(a, action_one_hot)
    # Void                        {action_one_hot, CNN1, observation_image}
    g = AddToLoss(f)
    program = Program([a, b, c, d, e], [f, g], 
        program_inputs, data_structures, optimizers, -1, "2_96_1")
    # -----------------

    return program, program_inputs, data_structures, optimizers

def build_2_84_0():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]
    
    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 38496
    # {'mean_performance': 348.8939943019943, 'mean_performance_std': 7.500704892685409}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {CNN1, new_observation_image}
    b = CNN(new_observation_image, CNN1)
    # FeatureVector32             {CNN1, FCN_32_to_32, observation_image, new_observation_image}
    c = PredictFeatureVector32FromFeatureVector32(a, b, FCN_32_to_32)
    # ListFeatureVector32         {observation_image, new_observation_image, VariableBuffer, CNN1, FCN_32_to_32}
    d = VariableAsBuffer(c, VariableBuffer)
    # FeatureVector32             {observation_image, new_observation_image, VariableBuffer, CNN1, FCN_32_to_32}
    e = MeanOfFeatureVectorList(d)
    # FeatureVectorActionSpace    {observation_image, new_observation_image, FCN_32_to_Action_Space, VariableBuffer, CNN1, FCN_32_to_32}
    f = FullyConnectedNetwork32toActionSpace(e, FCN_32_to_Action_Space)
    # RealNumber                  {observation_image, new_observation_image, FCN_32_to_Action_Space, CNN1, action_one_hot, VariableBuffer, FCN_32_to_32}
    g = DotProduct(action_one_hot, f)
    # update only:
    program = Program([a, b, c, d, e, f, g], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_0")
    # -----------------

    return program, program_inputs, data_structures, optimizers


def build_2_84_1():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()

    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]
    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 38491
    # {'mean_performance': 341.75630769230764, 'mean_performance_std': 3.2309705944574194}
    # ------
    a = CNN(observation_image, CNN1)                             # FeatureVector32             {CNN1, observation_image}
    b = CNN(new_observation_image, CNN1)                         # FeatureVector32             {new_observation_image, CNN1}
    c = PredictFeatureVector32FromFeatureVector32(a, b, FCN_32_to_32) # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, observation_image}
    d = VariableAsBuffer(c, VariableBuffer)                      # ListFeatureVector32         {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    e = ListVariance(d)                                          # NonNegativeNumber           {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    # update only: 
    f = SoftmaxAndNLL(a, action_one_hot)                         # NonNegativeNumber           {CNN1, action_one_hot, observation_image}
    g = AddToLoss(f)                                             # Void                        {CNN1, action_one_hot, observation_image}
    program = Program([a, b, c, d, e], [f, g], 
        program_inputs, data_structures, optimizers, -1, "2_84_1")
    # -----------------


    return program, program_inputs, data_structures, optimizers

def build_2_84_2():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]


    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 38493
    # {'mean_performance': 335.8520051282051, 'mean_performance_std': 2.891515710575494}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN1}
    b = CNN(new_observation_image, CNN1)
    # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, observation_image}
    c = PredictFeatureVector32FromFeatureVector32(a, b, FCN_32_to_32)
    # ListFeatureVector32         {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    d = VariableAsBuffer(c, VariableBuffer)
    # FeatureVector32             {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    e = MeanOfFeatureVectorList(d)
    # FeatureVectorActionSpace    {FCN_32_to_Action_Space, FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    f = FullyConnectedNetwork32toActionSpace(e, FCN_32_to_Action_Space)
    # NonNegativeNumber           {FCN_32_to_Action_Space, FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    g = L2Norm(f)
    # update only:
    program = Program([a, b, c, d, e, f, g], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_2")
    # -----------------


    return program, program_inputs, data_structures, optimizers

def build_2_84_3():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]
    
    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 38492
    # {'mean_performance': 332.9722358974359, 'mean_performance_std': 3.997428458442114}
    # ------
    a = CNN(observation_image, CNN1)                             # FeatureVector32             {CNN1, observation_image}
    b = CNN(new_observation_image, CNN1)                         # FeatureVector32             {new_observation_image, CNN1}
    c = PredictFeatureVector32FromFeatureVector32(a, b, FCN_32_to_32) # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, observation_image}
    d = VariableAsBuffer(c, VariableBuffer)                      # ListFeatureVector32         {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    e = MeanOfFeatureVectorList(d)                               # FeatureVector32             {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, CNN1}
    f = NearestNeighborRegressor(e, b, NearestNeighbor)          # FeatureVector32             {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, NearestNeighbor, CNN1}
    g = L2Norm(f)                                                # NonNegativeNumber           {FCN_32_to_32, observation_image, VariableBuffer, new_observation_image, NearestNeighbor, CNN1}
    # update only: 
    program = Program([a, b, c, d, e, f, g], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_3")
    # -----------------


    return program, program_inputs, data_structures, optimizers

def build_2_84_4():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]
    
    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 34198
    # {'mean_performance': 330.2381566951567, 'mean_performance_std': 4.37912656650276}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN2}
    b = CNN(new_observation_image, CNN2)
    # FeatureVector32             {observation_image, CNN1, new_observation_image, CNN2}
    c = AddFeatureVector(b, a)
    # FeatureVector32             {observation_image, CNN2, new_observation_image, CNN1, NearestNeighbor}
    d = NearestNeighborRegressor(c, a, NearestNeighbor)
    # FeatureVectorActionSpace    {FCN_32_to_Action_Space, observation_image, CNN2, new_observation_image, NearestNeighbor, CNN1}
    e = FullyConnectedNetwork32toActionSpace(d, FCN_32_to_Action_Space)
    # NonNegativeNumber           {FCN_32_to_Action_Space, action_one_hot, observation_image, CNN2, new_observation_image, CNN1, NearestNeighbor}
    f = L2Distance(action_one_hot, e)
    # update only:
    # Void                        {FCN_32_to_Action_Space, action_one_hot, observation_image, CNN2, new_observation_image, CNN1, NearestNeighbor}
    g = AddToLoss(f)
    program = Program([a, b, c, d, e, f], [g], 
        program_inputs, data_structures, optimizers, -1, "2_84_4")
    # -----------------


    return program, program_inputs, data_structures, optimizers

def build_2_84_5():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]


    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 38485
    # {'mean_performance': 285.03437916438605, 'mean_performance_std': 17.983580858348006}
    # ------
    # FeatureVector32             {CNN1, observation_image}
    a = CNN(observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN1}
    b = CNN(new_observation_image, CNN1)
    # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, observation_image}
    c = PredictFeatureVector32FromFeatureVector32(b, a, FCN_32_to_32)
    # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, observation_image}
    d = DetachFeatureVector(c)
    # FeatureVectorActionSpace    {FCN_32_to_Action_Space, FCN_32_to_32, observation_image, new_observation_image, CNN1}
    e = FullyConnectedNetwork32toActionSpace(d, FCN_32_to_Action_Space)
    # NonNegativeNumber           {FCN_32_to_Action_Space, action_one_hot, FCN_32_to_32, observation_image, new_observation_image, CNN1}
    f = L2Distance(action_one_hot, e)
    # update only:
    # Void                        {FCN_32_to_Action_Space, action_one_hot, FCN_32_to_32, observation_image, new_observation_image, CNN1}
    g = AddToLoss(f)
    program = Program([a, b, c, d, e, f], [g], 
        program_inputs, data_structures, optimizers, -1, "2_84_5")
    # -----------------

    return program, program_inputs, data_structures, optimizers

def build_2_84_6():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 10837
    # {'mean_performance': 281.84870158159725, 'mean_performance_std': 20.952152741848312}
    # ------
    # FeatureVector32             {FCN_Action_Space_to_32, action_one_hot}
    a = FullyConnectedNetworkActionSpaceto32(
        action_one_hot, FCN_Action_Space_to_32)
    # FeatureVector32             {new_observation_image, CNN1}
    b = CNN(new_observation_image, CNN1)
    # FeatureVector32             {action_one_hot, FCN_Action_Space_to_32, new_observation_image, NearestNeighbor, CNN1}
    c = NearestNeighborRegressor(b, a, NearestNeighbor)
    # FeatureVectorActionSpace    {action_one_hot, FCN_64_to_Action_Space, FCN_Action_Space_to_32, new_observation_image, CNN1}
    d = FullyConnectedNetworkTwo32toActionSpace(b, a, FCN_64_to_ACTION_SPACE)
    # FeatureVectorActionSpace    {FCN_32_to_Action_Space, action_one_hot, FCN_Action_Space_to_32, new_observation_image, NearestNeighbor, CNN1}
    e = FullyConnectedNetwork32toActionSpace(c, FCN_32_to_Action_Space)
    # RealNumber                  {FCN_32_to_Action_Space, FCN_Action_Space_to_32, FCN_64_to_Action_Space, new_observation_image, CNN1, NearestNeighbor, action_one_hot}
    f = DotProduct(e, d)
    # update only:
    program = Program([a, b, c, d, e, f], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_6")
    # -----------------

    return program, program_inputs, data_structures, optimizers

def build_2_84_7():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 48248
    # {'mean_performance': 269.2592013043933, 'mean_performance_std': 11.728057478637911}
    # ------
    # FeatureVector32             {new_observation_image, CNN1}
    a = CNN(new_observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN2}
    b = CNN(new_observation_image, CNN2)
    # FeatureVectorActionSpace    {FCN_32_to_Action_Space, new_observation_image, CNN2}
    c = FullyConnectedNetwork32toActionSpace(b, FCN_32_to_Action_Space)
    # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, CNN2}
    d = PredictFeatureVector32FromFeatureVector32(b, a, FCN_32_to_32)
    # ListFeatureVector32         {FCN_32_to_Action_Space, Ensemble_32_Action_to_32_1, FCN_32_to_32, CNN2, new_observation_image, CNN1}
    e = FullyConnectedNetworkEnsemble32AndActionTo32(
        d, c, Ensemble_32_Action_to_32_1)
    # FeatureVector32             {FCN_32_to_Action_Space, Ensemble_32_Action_to_32_1, FCN_32_to_32, CNN2, new_observation_image, CNN1}
    f = MeanOfFeatureVectorList(e)
    # NonNegativeNumber           {FCN_32_to_Action_Space, action_one_hot, Ensemble_32_Action_to_32_1, FCN_32_to_32, CNN2, new_observation_image, CNN1}
    g = SoftmaxAndNLL(f, action_one_hot)
    # update only:
    program = Program([a, b, c, d, e, f, g], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_7")
    # -----------------

    return program, program_inputs, data_structures, optimizers

def build_2_84_8():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]


    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 45415
    # {'mean_performance': 266.4210974358974, 'mean_performance_std': 4.477815148527007}
    # ------
    # FeatureVector32             {new_observation_image, CNN1}
    a = CNN(new_observation_image, CNN1)
    # FeatureVector32             {new_observation_image, CNN2}
    b = CNN(new_observation_image, CNN2)
    # ListFeatureVector32         {new_observation_image, CNN1, VariableBuffer}
    c = VariableAsBuffer(a, VariableBuffer)
    # FeatureVector32             {CNN1, FCN_32_to_32, new_observation_image, CNN2}
    d = PredictFeatureVector32FromFeatureVector32(a, b, FCN_32_to_32)
    # FeatureVector32             {FCN_32_to_32, CNN2, new_observation_image, NearestNeighbor, CNN1}
    e = NearestNeighborRegressor(b, d, NearestNeighbor)
    # ListFeatureVector32         {FCN_32_to_32, CNN2, VariableBuffer, new_observation_image, NearestNeighbor, CNN1}
    f = FeatureVectorListMinusFeatureVector(c, e)
    # NonNegativeNumber           {FCN_32_to_32, CNN2, VariableBuffer, new_observation_image, NearestNeighbor, CNN1}
    g = ListVariance(f)
    # update only:
    program = Program([a, b, c, d, e, f, g], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_8")
    # -----------------


    return program, program_inputs, data_structures, optimizers

def build_2_84_9():
    program_inputs, data_structures, optimizers = get_2_84_datastructures()
    
    observation_image, action_one_hot, new_observation_image = program_inputs
    FCN_64_to_32, CNN1, FCN_32_to_Action_Space, FCN_Action_Space_to_32, \
        FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1, FCN_32_to_32, \
        RunningNormData, VariableBuffer, FeatureVectorRunningNormData, NearestNeighbor, CNN2, Ensemble_32_Action_to_32_1 = data_structures
    Adam = optimizers[0]

    # -----------------
    # Experiment: 2-84_15x15_new-ppo-real-batched-shared_2500-steps_5-trials
    # Index 48208
    # {'mean_performance': 254.91209401709403, 'mean_performance_std': 8.261126411381378}
    # ------
    a = CNN(new_observation_image, CNN1)                         # FeatureVector32             {new_observation_image, CNN1}
    b = CNN(new_observation_image, CNN2)                         # FeatureVector32             {new_observation_image, CNN2}
    c = FullyConnectedNetwork32toActionSpace(b, FCN_32_to_Action_Space) # FeatureVectorActionSpace    {FCN_32_to_Action_Space, new_observation_image, CNN2}
    d = FullyConnectedNetworkActionSpaceto32(c, FCN_Action_Space_to_32) # FeatureVector32             {FCN_Action_Space_to_32, new_observation_image, FCN_32_to_Action_Space, CNN2}
    e = AddFeatureVector(a, d)                                   # FeatureVector32             {FCN_32_to_Action_Space, FCN_Action_Space_to_32, CNN2, new_observation_image, CNN1}
    f = PredictFeatureVector32FromFeatureVector32(a, e, FCN_32_to_32) # FeatureVector32             {FCN_32_to_Action_Space, FCN_32_to_32, FCN_Action_Space_to_32, CNN2, new_observation_image, CNN1}
    g = L2Norm(f)                                                # NonNegativeNumber           {FCN_32_to_Action_Space, FCN_32_to_32, FCN_Action_Space_to_32, CNN2, new_observation_image, CNN1}
    # update only: 
    program = Program([a, b, c, d, e, f, g], [], 
        program_inputs, data_structures, optimizers, -1, "2_84_9")
    # -----------------

    return program, program_inputs, data_structures, optimizers

def get_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image")

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    FCN_64_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32(), 3, "FCN_64_to_32", True)
    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True)
    FCN_32_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace(), 5, "FCN_32_to_ACTION_SPACE", True)
    FCN_Action_Space_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(), 6, "FCN_Action_Space_to_32", True)
    FCN_64_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(), 7, "FCN_64_to_ACTION_SPACE", True)
    ENSEMBLE_CNN1 = Variable(
        program_types.EnsembleWeightsImageTo32(), 8, "ENSEMBLE_CNN1", True)
    EnsembleFCN1 = Variable(
        program_types.EnsembleWeights32To32(), 9, "ENSEMBLE_CNN1", True)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeights32AndActionTo32(), 9)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeightsTwo32To32(), 9)
    data_structures = [FCN_64_to_32, CNN1, FCN_32_to_ACTION_SPACE, FCN_Action_Space_to_32,
                       FCN_64_to_ACTION_SPACE, ENSEMBLE_CNN1, EnsembleFCN1]

    Adam = Variable(program_types.AdamOptimizer(), 10, "Adam")
    optimizers = [Adam]

    # a = ActionToOneHotFeatureVector(action)                      # FeatureVectorActionSpace    {action}
    # ListFeatureVector32         {observation_image, EnsembleCNN1}
    b = CNNEnsemble(observation_image, ENSEMBLE_CNN1)
    c = MeanOfFeatureVectorList(b)                               # FeatureVector32             {observation_image, EnsembleCNN1}
    d = DetachFeatureVector(c)                                   # FeatureVector32             {observation_image, EnsembleCNN1}
    # FeatureVectorActionSpace    {observation_image, FCN_32_to_Action_Space, EnsembleCNN1}
    e = FullyConnectedNetwork32toActionSpace(d, FCN_32_to_ACTION_SPACE)
    f = DetachFeatureVector(e)                                   # FeatureVectorActionSpace    {observation_image, FCN_32_to_Action_Space, EnsembleCNN1}
    g = L2Distance(action_one_hot, f)                                         # NonNegativeNumber           {observation_image, FCN_32_to_Action_Space, action, EnsembleCNN1}
    # update only: 
    h = L2Distance(action_one_hot, e)                                         # NonNegativeNumber           {observation_image, FCN_32_to_Action_Space, action, EnsembleCNN1}
    i = AddToLoss(h)                                   # Void                        {observation_image, FCN_32_to_Action_Space, Adam, action, EnsembleCNN1}
    program = Program([action_one_hot, b, c, d, e, f, g], [h, i], 
        program_inputs, data_structures, optimizers, -1, "get")

    # -----------------

    return program, program_inputs, data_structures, optimizers

def build_inverse_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image",
        short_name="s(t)")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot",
        short_name="a(t)")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image",
        short_name="s(t+1)")

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    adam = Variable(program_types.AdamOptimizer(), 5, "Adam")
    optimizers = [adam]

    cnn_action_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(), 6, "FCN_Action_Space_to_32", True, 
        short_name="θ{1}: 𝔸 → 𝔽")

    cnn_64_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(), 7, "FCN_64_to_ACTION_SPACE", True,
        short_name="θ{2}: 𝔽 x 𝔽 → 𝔸")

    cnn_64_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32(), 8, "FCN_64_to_32", True,
        short_name="θ{3}: 𝔽 x 𝔽 → 𝔽")

    cnn_state_encoder = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 9, "CNN", True,
        short_name="θ{4}: 𝕊 → 𝔽")

    data_structures = [cnn_action_to_32, cnn_64_to_32,
                       cnn_64_to_ACTION_SPACE, cnn_state_encoder]

    action_32 = FullyConnectedNetworkActionSpaceto32(
        action_one_hot, cnn_action_to_32)
    encoded_state = CNN(observation_image, cnn_state_encoder)
    predicted_new_state = FullyConnectedNetworkTwo32to32(
        action_32, encoded_state, cnn_64_to_32)
    encoded_new_state = CNN(
        new_observation_image, cnn_state_encoder)
    error_in_new_state_prediction = L2Distance(
        encoded_new_state, predicted_new_state)

    predicted_action = FullyConnectedNetworkTwo32toActionSpace(
        encoded_state, encoded_new_state, cnn_64_to_ACTION_SPACE)
    # predicted_action_i = FullyConnectedNetworkTwo32to32(
    #     encoded_state, encoded_new_state)
    # predicted_action = FullyConnectedNetwork32toActionSpace(predicted_action_i)
    error_in_action_prediction = SoftmaxAndNLL(
        predicted_action, action_one_hot)
    error = Add(
        error_in_action_prediction, error_in_new_state_prediction)
    m = AddToLoss(error)

    program = Program(
        [action_32, encoded_state, predicted_new_state,
            encoded_new_state, error_in_new_state_prediction],
        [predicted_action, error_in_action_prediction, error, m],
        program_inputs, data_structures, optimizers, -1, "inverse"
    )

    return program, program_inputs, data_structures, optimizers


def build_disagreement_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image", 
        short_name="s(t)")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot", 
        short_name="a(t)")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image", 
        short_name="s(t+1)")

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    cnn = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 3, "CNN1", True,
        short_name="θ{1}: 𝕊 → 𝔽")
    # cnn = Variable(
    #     program_types.NeuralNetworkWeightsFeatureVectorObservationSpaceToFeatureVector32(), 3)
    ensemble = Variable(
        program_types.EnsembleWeights32AndActionTo32(), 4, "ENSEMBLE", True, 
        short_name="θ{2}: 𝔽 x 𝔸 → [𝔽]")
    data_structures = [cnn, ensemble]

    adam = Variable(program_types.AdamOptimizer(), 5, "Adam", True)
    optimizers = [adam]

    b = CNNWithoutGradients(observation_image, cnn)
    c = FullyConnectedNetworkEnsemble32AndActionTo32(
        b, action_one_hot, ensemble)
    d = ListVariance(c)

    f = CNNWithoutGradients(new_observation_image, cnn)
    i = FeatureVectorListAverageL2DistanceToFeatureVector(c, f)
    j = AddToLoss(i)

    program = Program(
        [b, c, d], [f, i, j],  
        program_inputs, data_structures, optimizers,
        -1, "disagreement"
    )

    return program, program_inputs, data_structures, optimizers

def build_random_network_distillation_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image")
    action_one_hot = Variable(program_types.FeatureVectorActionSpace(), 1, "action_one_hot")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image",
        short_name="s(t+1)")

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    FCN_64_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVector32(), 3, "FCN_64_to_32", True)
    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True,
        short_name="θ{1}: 𝕊 → 𝔽")
    CNN2 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 11, "CNN2", True,
        short_name="θ{2}: 𝕊 → 𝔽")
    FCN_32_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector32ToFeatureVectorActionSpace(), 5, "FCN_32_to_ACTION_SPACE", True)
    FCN_Action_Space_to_32 = Variable(
        program_types.NeuralNetworkWeightsFeatureVectorActionSpaceToFeatureVector32(), 6, "FCN_Action_Space_to_32", True)
    FCN_64_to_ACTION_SPACE = Variable(
        program_types.NeuralNetworkWeightsFeatureVector64ToFeatureVectorActionSpace(), 7, "FCN_64_to_ACTION_SPACE", True)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeights32AndActionTo32(), 9)
    # EnsembleFCN1 = Variable(
    #     program_types.EnsembleWeightsTwo32To32(), 9)
    data_structures = [FCN_64_to_32, CNN1, CNN2, FCN_32_to_ACTION_SPACE, FCN_Action_Space_to_32,
                       FCN_64_to_ACTION_SPACE]

    Adam = Variable(program_types.AdamOptimizer(), 10, "Adam")
    optimizers = [Adam]

    random_state_encoding = CNNWithoutGradients(new_observation_image, CNN1)
    learned_state_encoding = CNN(new_observation_image, CNN2)
    err = L2Distance(random_state_encoding, learned_state_encoding)

    m = AddToLoss(err)

    program = Program(
        [random_state_encoding, learned_state_encoding, err],
        [m],
        program_inputs, data_structures, optimizers, -1, "rnd"
    )

    return program, program_inputs, data_structures, optimizers


def build_random_noise_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image")

    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True)

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    data_structures = [CNN1]

    Adam = Variable(program_types.AdamOptimizer(), 10)
    optimizers = [Adam]

    r = NormalDistribution()

    program = Program([r], [],
        program_inputs, data_structures, optimizers, -1, "noise"
    
    )

    return program, program_inputs, data_structures, optimizers

def build_constant_zero_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image")

    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True)

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    data_structures = [CNN1]

    Adam = Variable(program_types.AdamOptimizer(), 10)
    optimizers = [Adam]

    r = ConstantZero()

    program = Program([r], [],
        program_inputs, data_structures, optimizers, -1, "zero"
    )

    return program, program_inputs, data_structures, optimizers

def build_constant_negative_one_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image")

    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True)

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    data_structures = [CNN1]

    Adam = Variable(program_types.AdamOptimizer(), 10)
    optimizers = [Adam]

    r = ConstantNegativeOne()

    program = Program([r], [],
        program_inputs, data_structures, optimizers, -1, "negative_one"
    )

    return program, program_inputs, data_structures, optimizers

def build_constant_one_program():
    observation_image = Variable(
        program_types.ImageTensor(), 0, "observation_image")
    action_one_hot = Variable(
        program_types.FeatureVectorActionSpace(), 1, "action_one_hot")
    new_observation_image = Variable(
        program_types.ImageTensor(), 2, "new_observation_image")

    CNN1 = Variable(
        program_types.NeuralNetworkWeightsObservationToFeatureVector32(), 4, "CNN1", True)

    program_inputs = [observation_image, action_one_hot, new_observation_image]

    data_structures = [CNN1]

    Adam = Variable(program_types.AdamOptimizer(), 10)
    optimizers = [Adam]

    r = ConstantOne()

    program = Program([r], [],
        program_inputs, data_structures, optimizers, -1, "one"
    )

    return program, program_inputs, data_structures, optimizers

if __name__ == "__main__":
    main()
