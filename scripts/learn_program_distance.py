
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from typing import Union, List
import functools
from tqdm import tqdm
import os

from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import mlca.operations as operations
from mlca.operations_list import OperationsSet, OperationsSetList
import mlca.helpers.config
import mlca.helpers.probability
from mlca.program import Program, ProgramId
from mlca.scripts.analyze_synthesized_programs import load_program_data, programs_by_mean_performance, ProgramData
from mlca.test_synthesized_programs_experiments import TspParams, TspExperimentList
from mlca.search_program_experiments import SearchParams
from mlca.run_agent import Reward
from mlca.predict_performance import program_as_feature_vector_learned_program_distance
from mlca.learn_program_distance_experiments import ProgramDistanceExperimentList, ProgramDistanceParams

ProgramDistance = float

def main(experiment_id=None):
    if experiment_id is None:
        parser = mlca.helpers.config.argparser()
        args = parser.parse_args()
        experiment_id = args.experiment_id

        program_distance_params = ProgramDistanceExperimentList[experiment_id]
        tsp_params = TspExperimentList.get(program_distance_params.TEST_SYNTHESIZED_PROGRAMS_EXP_NAME)

        with program_distance_params:
            with tsp_params:
                print(experiment_id, ProgramDistanceParams.current().TEST_SYNTHESIZED_PROGRAMS_EXP_NAME)

                # =======================
                # Load data
                # =======================
                data, _, curiosity_programs, _,  _, _,  _, _,  _, _,  _ = load_program_data(
                    ProgramDistanceParams.current().TEST_SYNTHESIZED_PROGRAMS_EXP_NAME)
                random.seed(3)
                random.shuffle(data)

                # data = data[:1000]
                # print("WARNING FIRST 1000")

                program_id_to_program = {p.program_id: p for p in curiosity_programs}
                program_id_to_data = {d.curiosity_program.program_id: d for d in data}

                # =======================
                # Convert data to regressor input
                # =======================
                program_correlations: List[ProgramDistance] = []
                program_ids_a: List[ProgramId] = []
                program_ids_b: List[ProgramId] = []
                for d in data:
                    for a, program_a_id in enumerate(d.results.program_correlation_ids):
                        for b, program_b_id in enumerate(d.results.program_correlation_ids):
                            if b > a:
                                correlation = np.mean([t[a][b] for t in d.results.trial_program_correlations])
                                program_ids_a.append(program_a_id)
                                program_ids_b.append(program_b_id)
                                program_correlations.append(correlation)
                                if correlation < -.9995: # and program_a_id in program_id_to_data and program_b_id in program_id_to_data:
                                    # print(program_id_to_data[program_a_id].stats["mean_performance"])
                                    # print(program_id_to_data[program_b_id].stats["mean_performance"])
                                    program_id_to_program[program_a_id].visualize_as_graph("A")
                                    program_id_to_program[program_b_id].visualize_as_graph("B")
                                    quit()

                # print(program_correlations)

                print("smallest program correlations", sorted(program_correlations)[: 10])

                plt.hist(program_correlations, bins=50)
                plt.show()

                assert len(program_ids_b) == len(program_ids_a) and len(program_ids_a) == len(program_correlations)
                print(f"# evaluated programs {len(data)}")
                print(f"# datapoints {len(program_correlations)}")

                all_program_ids = set(program_ids_a + program_ids_b)
                print(f"# all_program_ids {len(all_program_ids)}")

                program_id_to_feature_vector = {
                    program_id: program_as_feature_vector_learned_program_distance(
                        program_id_to_program[program_id]
                    )
                    for program_id in tqdm(all_program_ids, "Convert programs to feature vectors")
                }  

                def make_regressor_input(fv_a, fv_b):
                    return np.concatenate((fv_a, fv_b))

                X = np.array([
                    make_regressor_input(
                        program_id_to_feature_vector[program_a_id],
                        program_id_to_feature_vector[program_b_id]
                    )
                    for program_a_id, program_b_id
                    in zip(program_ids_a, program_ids_b)])
                y = np.array([
                    c 
                    for c in program_correlations])
                y_mean = y.mean()
                y_std = (y - y_mean).std()
                y = (y - y_mean) / y_std
                def prediction_to_correlation(prediction):
                    return prediction * y_std + y_mean

                # =======================
                # Partition data
                # =======================
                num_train = int(len(y) * .8)
                train_X, train_y = X[:num_train], y[:num_train]
                test_X, test_y = X[num_train:], y[num_train:]

                # =======================
                # Run regressor
                # =======================
                regr = MLPRegressor((100,100), early_stopping=True)
                # regr = MLPRegressor()
                regr.fit(train_X, train_y)
                train_y_pred = regr.predict(train_X)
                test_y_pred = regr.predict(test_X)
                print("Loss", regr.loss)
                print("Train Score", regr.score(train_X, train_y))
                print("Test Score", regr.score(test_X, test_y))
                print("Train MSE", mean_squared_error(train_y_pred, train_y))
                print("Test MSE", mean_squared_error(test_y_pred, test_y))
                # print(regr.get_params())
                # print(regr.coefs_)
                # print(regr.intercepts_)

                # ======================
                # Visualize correlation of top 10 programs
                # =======================
                ORIG_DATA_EXP_NAME = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"
                orig_data, _, _, _,  _, _,  _, _,  _, _,  _ = load_program_data(
                    ORIG_DATA_EXP_NAME)

                NUM_BEST = 16   
                best = programs_by_mean_performance(orig_data)
                best = list(reversed(best[-NUM_BEST:]))
                print([d.curiosity_program.program_id for d in best])
                inps = []
                for a in range(NUM_BEST):
                    for b in range(a + 1, NUM_BEST):
                        inps.append(make_regressor_input(
                            program_as_feature_vector_learned_program_distance(
                                best[a].curiosity_program
                            ),
                            program_as_feature_vector_learned_program_distance(
                                best[b].curiosity_program
                            )
                        ))
                correlations = prediction_to_correlation(regr.predict(inps))
                correlations_matrix = np.ones((NUM_BEST, NUM_BEST))
                i = 0
                for a in range(NUM_BEST):
                    for b in range(a + 1, NUM_BEST):
                        correlations_matrix[a][b] = correlations[i]
                        correlations_matrix[b][a] = correlations[i]
                        i += 1

                for x in correlations_matrix:
                    print(x)

                for i, d in enumerate(best):
                    d.curiosity_program.visualize_as_graph(i)

                # ======================
                # Visualize regressor
                # =======================
                # inps = []
                # all_program_ids = list(all_program_ids)
                # for i in range(10000):
                #     a = random.choice(all_program_ids)
                #     b = random.choice(all_program_ids)
                #     inps.append(make_regressor_input(
                #         program_id_to_feature_vector[a],
                #         program_id_to_feature_vector[b]
                #     ))
                # correlations = regr.predict(inps)
                # print(correlations)
                # import matplotlib.pyplot as plt
                # plt.hist(correlations)
                # plt.show()



if __name__ == "__main__":
  main()
