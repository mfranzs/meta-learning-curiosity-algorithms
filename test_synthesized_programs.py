"""
Helper code that loads up a program search. Interfaces with search_programs.py
to run the search.
"""

import torch
import random
import os

import mlca.helpers.config
import mlca.helpers.debug
import mlca.helpers.util

from mlca.test_synthesized_programs_experiments import TspParams, TspExperimentList
from mlca.simulate_search import _select_next_program_batch_random, _select_next_program_preprocess_data_diversity, _select_next_program_batch_diversity, _none, _select_next_program_batch_regressor, get_early_termination_batch_data_fn, rollout_timestep_pruning_hook_fn
from mlca.search_programs import search_with_score_prediction, _select_next_program_data_update_with_program_result_diversity
from mlca.search_program_experiments import SearchParams, SearchExperimentList
from mlca.scripts.analyze_synthesized_programs import load_curiosity_program_data
from mlca.run_agent import evaluate_program_in_environment

def test_all_curiosity_programs(parser, args):
    NUM_WORKERS = args.workers_per_gpu * args.num_gpus

    evaluation_folder = f"pickles/{args.experiment_id}_evaluations/"
    curiosity_programs_name = TspParams.current().CURIOSITY_PROGRAMS_NAME
    reward_combiner_programs_name = TspParams.current().REWARD_COMBINER_PROGRAMS_NAME

    if not os.path.exists(evaluation_folder):
        os.mkdir(evaluation_folder)

    # Initialize the GPUs here to prevent multiprocessing problems
    print("Initialize GPUS")
    # https://github.com/pytorch/pytorch/issues/16559
    if torch.cuda.is_available():
        for i in range(args.num_gpus):
            with torch.cuda.device(i):
                torch.tensor([1.]).cuda()

    pre_evaluated_data, _, \
        curiosity_programs, curiosity_program_inputs, \
        curiosity_data_structures, curiosity_optimizers, \
        reward_combiner_programs, reward_combiner_program_inputs, \
        reward_combiner_data_structures, reward_combiner_optimizers, _ = load_curiosity_program_data(
            curiosity_programs_name, reward_combiner_programs_name, args.experiment_id, 
            TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID)

    restricted_programs_to_evaluate = TspParams.current().RESTRICTED_CURIOSITY_PROGRAMS_TO_EVALUATE

    if TspParams.current().SPLIT_ACROSS_MACHINES:
        assert args.machine_split_id >= 0 and args.machine_split_id < TspParams.current().SPLIT_ACROSS_MACHINES
        if restricted_programs_to_evaluate is None:
            random.seed(0)
            random_curiosity_programs = list(curiosity_programs)
            random.shuffle(random_curiosity_programs)
            restricted_programs_to_evaluate = [
                p.program_id for i, p in enumerate(random_curiosity_programs) 
                if i % TspParams.current().SPLIT_ACROSS_MACHINES == args.machine_split_id
            ]
        else:
            random.seed(0)
            random_curiosity_program_ids = list(restricted_programs_to_evaluate)
            random.shuffle(random_curiosity_program_ids)
            restricted_programs_to_evaluate = [
                p_id for i, p_id in enumerate(random_curiosity_program_ids)
                if i % TspParams.current().SPLIT_ACROSS_MACHINES == args.machine_split_id
            ]
    else:
        assert args.machine_split_id == None

    reward_combiner_program = reward_combiner_programs[TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID] \
        if TspParams.current().FIXED_REWARD_COMBINER_PROGRAM_ID else None

    print("reward_combiner_program", reward_combiner_program)

    if restricted_programs_to_evaluate:
        print("restricted_programs_to_evaluate", len(restricted_programs_to_evaluate), restricted_programs_to_evaluate[:10], "... etc ...")
        restricted_programs_to_evaluate_set = set(restricted_programs_to_evaluate)

    restricted_programs = [
        p \
        for p in curiosity_programs
        if not restricted_programs_to_evaluate or p.program_id in restricted_programs_to_evaluate_set
    ]
    assert not restricted_programs_to_evaluate or len(restricted_programs) == len(restricted_programs_to_evaluate)

    id_to_program = { p.program_id: p for p in restricted_programs }

    def get_pre_evaluated_programs_fn():
        progs = [
            id_to_program[d.results.curiosity_program_id] 
            for d in pre_evaluated_data 
            if restricted_programs_to_evaluate is None \
                or d.results.curiosity_program_id in restricted_programs_to_evaluate]
        print("Get pre evaluated results. # Datapoints:", len(progs), len(pre_evaluated_data))
        return progs, pre_evaluated_data

    def post_batch_hook_fn(a):
        pass

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


    search_with_score_prediction(
        restricted_programs, 
        get_pre_evaluated_programs_fn, args.num_gpus * args.workers_per_gpu,
        evaluate_program_in_environment, rollout_timestep_pruning_hook_fn, 
        select_next_program_batch_fn, select_next_program_preprocess_data_fn,
        select_next_program_data_update_with_program_result_fn, post_batch_hook_fn, 
        get_early_termination_batch_data_fn, simulate_params, params, 
        (args.num_gpus, reward_combiner_program, evaluation_folder)
    )

# def test_all_reward_combiner_programs(parser, args):
#     NUM_WORKERS = args.workers_per_gpu * args.num_gpus    

#     mp.set_start_method('spawn')

#     evaluation_folder = f"pickles/{args.experiment_id}_evaluations/"
#     curiosity_programs_name = TspParams.current().CURIOSITY_PROGRAMS_NAME
#     reward_combiner_programs_name = TspParams.current().REWARD_COMBINER_PROGRAMS_NAME

#     if not os.path.exists(evaluation_folder):
#         os.mkdir(evaluation_folder)

#     pre_evaluated_data, _, \
#         curiosity_programs, curiosity_program_inputs, \
#         curiosity_data_structures, curiosity_optimizers, \
#         reward_combiner_programs, reward_combiner_program_inputs, \
#         reward_combiner_data_structures, reward_combiner_optimizers, _ = load_reward_combiner_program_data(
#         curiosity_programs_name, reward_combiner_programs_name, args.experiment_id, TspParams.current().FIXED_CURIOSITY_PROGRAM_ID)

#     curiosity_program = curiosity_programs[TspParams.current().FIXED_CURIOSITY_PROGRAM_ID]
#     print("Fixed curiosity program: ")
#     print(pretty_program_p(curiosity_program))

#     restricted_programs_to_evaluate = TspParams.current()(
#         "RESTRICTED_REWARD_COMBINER_PROGRAMS_TO_EVALUATE", None)

#     pre_evaluated_program_ids = set(d.reward_combiner_program.program_id for d in pre_evaluated_data)

#     restricted_programs = [
#         ProgramWithId(p.forward_program, p.update_program, program_id)
#         for program_id, p in enumerate(reward_combiner_programs)
#         if (not restricted_programs_to_evaluate or program_id in restricted_programs_to_evaluate) and program_id not in pre_evaluated_program_ids
#     ]

#     @threadpool
#     def evaluate_program_batch_fn(
#         program_batch, rollout_timestep_pruning_hook_fn):

#         worker_args = []
#         for _, reward_combiner_program in enumerate(program_batch):
#             worker_args.append((
#                 TspParams.current().FIXED_CURIOSITY_PROGRAM_ID, 
#                 reward_combiner_program.program_id, 
#                 params, 
#                 args.num_gpus,
#                 curiosity_program, curiosity_program_inputs, curiosity_data_structures, curiosity_optimizers,
#                 reward_combiner_program, reward_combiner_program_inputs, reward_combiner_data_structures, reward_combiner_optimizers,
#                 evaluation_folder, False))

#         results_data = pool.map(
#             evaluate_program_in_environment, worker_args)
#         results_data = [_stats_for_program(r) for r in results_data]

#         pool.close()
#         pool.terminate()

#         return results_data

#     # Search through the rest of the unevaluated programs
#     BATCH_SIZE = 1000
#     for batch_start in range(0, len(restricted_programs), BATCH_SIZE):
#         program_batch = restricted_programs[batch_start: batch_start + BATCH_SIZE]

#         print(f"Evaluated {len(pre_evaluated_data)} programs. Current batch has {len(program_batch)} programs.")

#         batch_results = evaluate_program_batch_fn(
#             program_batch, rollout_timestep_pruning_hook_fn)

if __name__ == "__main__":
    parser = mlca.helpers.config.argparser()
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--workers_per_gpu', default=4, type=int)
    parser.add_argument('--machine_split_id', default=None, type=int)
    args = parser.parse_args()

    params = TspExperimentList.get(args.experiment_id)
    simulate_params = SearchExperimentList.get(params.SEARCH_PROGRAMS_EXPERIMENT_ID)

    device = mlca.helpers.config.get_device_and_set_default()

    print("device", device)

    with params:
        with simulate_params:
            with mlca.helpers.config.DefaultDevice(device):
                if TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.CURIOSITY_SEARCH:
                    test_all_curiosity_programs(parser, args)
                elif TspParams.current().EXPERIMENT_TYPE == TspParams.ExperimentType.REWARD_COMBINER_SEARCH:
                    quit("Deprecated")
                    # test_all_reward_combiner_programs(parser, args)
                else: 
                    quit(TspParams.current().EXPERIMENT_TYPE)
