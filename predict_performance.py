"""
Predict a program's performance from the program's structure. Encode
each program into a feature vector, then predict new program performances.
"""

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

import mlca.operations as operations
from mlca.operations_list import OperationsSet, OperationsSetList
import mlca.helpers.config
import mlca.helpers.probability
from mlca.program import Program
from mlca.scripts.analyze_synthesized_programs import load_program_data, ProgramData
from mlca.predict_performance_experiments import PredictPerformanceExperimentList, PredictPerformanceParams
from mlca.test_synthesized_programs_experiments import TspParams, TspExperimentList
from mlca.search_program_experiments import SearchParams
from mlca.learn_program_distance_experiments import ProgramDistanceParams
from mlca.run_agent import Reward

ProgramFeatureVector = np.ndarray

# ===========================
# Bayesian Model
# ===========================
def prob_programs_above_perf_threshold_knn(
    program_feature_vectors: List[ProgramFeatureVector], 
    regressor: KNeighborsRegressor, 
    perf_threshold: Reward) -> List[float]:
  
  dist, ind = regressor.kneighbors(
    np.array(program_feature_vectors)
  )

  probs = [
    1 - mlca.helpers.probability.cdf(
      perf_threshold, 
      np.mean(regressor._y[ind[i]]), 
      np.std(regressor._y[ind[i]])
    )
    for i in tqdm(range(len(program_feature_vectors)), "prob_programs_above_perf_threshold")
  ]

  return probs

def prob_programs_above_perf_threshold_gp(
    program_feature_vectors: List[ProgramFeatureVector], 
    regressor: GaussianProcessRegressor, 
    perf_threshold: Reward) -> List[float]:

  pred_mean, pred_std = regressor.predict(
    np.array(program_feature_vectors),
    return_std=True
  )

  probs = [
    1 - mlca.helpers.probability.cdf(
      perf_threshold, 
      pred_mean[i], 
      pred_std[i]
    )
    for i in tqdm(range(len(program_feature_vectors)), "prob_programs_above_perf_threshold")
  ]

  return probs

# ===========================
# Original Model
# ===========================
def program_as_feature_vector_diversity(p: Program) -> ProgramFeatureVector:
  return _program_as_feature_vector(
    p, 
    SearchParams.current().DIVERSITY_FEATURE_PAIRS, 
    SearchParams.current().DIVERSITY_FEATURE_INPUT_OUTPUT)

def program_as_feature_vector_predict_performance(p: Program) -> ProgramFeatureVector:
  return _program_as_feature_vector(
    p, 
    PredictPerformanceParams.current().FEATURE_PAIRS, 
    PredictPerformanceParams.current().FEATURE_INPUT_OUTPUT)

def program_as_feature_vector_learned_program_distance(p: Program) -> ProgramFeatureVector:
  return _program_as_feature_vector(
    p, 
    ProgramDistanceParams.current().FEATURE_PAIRS, 
    ProgramDistanceParams.current().FEATURE_INPUT_OUTPUT)

def _program_as_feature_vector(p: Program, fp: int, feature_input_output: bool) -> ProgramFeatureVector:
  assert type(fp) == int, fp

  operations_list_name = TspParams.current().CURIOSITY_PROGRAMS_NAME
  ops = OperationsSetList[operations_list_name].OPERATIONS  

  FEATURE_VECTOR_CLASSES = [[o] for o in ops] + \
      (list(itertools.combinations(ops, 2)) if fp >= 2 else []) + \
      (list(itertools.combinations(ops, 3)) if fp >= 3 else [])
  FEATURE_VECTOR_INPUT_OUTPUT_PAIR_CLASSES = \
      (list(itertools.permutations(ops, 2))
      if feature_input_output else [])

  program_ops = p.forward_program + p.update_program
  classes = set([type(d) for d in program_ops])

  r1 = []
  for has_gradients in [True, False]:
    classes_and_grads = [(c, has_gradients) for c in classes]
    for ops in FEATURE_VECTOR_CLASSES:
      r1.append(all((o, has_gradients) in classes_and_grads for o in ops))

  r2 = []
  for in_class, out_class in FEATURE_VECTOR_INPUT_OUTPUT_PAIR_CLASSES:
    r2.append(
      any(
        (type(o) == out_class \
          and any(type(i) == in_class for i in o.inputs))
        for o \
        in program_ops))

  r = r1 + r2

  assert max(r1) == 1
  assert not feature_input_output or max(r2) == 1
  assert max(r) == 1

  return np.array(r)

def _get_target_output(d: ProgramData):
  return d.stats["mean_performance"]

def get_predict_performance_regressor():
  MODEL = PredictPerformanceParams.current().MODEL
  if MODEL == "Linear":
    regr = linear_model.LinearRegression()
  elif MODEL == "RandomForest":
    regr = RandomForestRegressor(n_estimators=PredictPerformanceParams.current().NUM_ESTIMATORS, max_depth=2,
                                 random_state=0)
  elif MODEL == "KNN":
    regr = KNeighborsRegressor(
        PredictPerformanceParams.current().NEIGHBORS, 
        "uniform", 
        algorithm="kd_tree", 
        metric="manhattan")
  elif MODEL == "GP":
    regr = GaussianProcessRegressor(
      normalize_y=True)
  return regr

def main(experiment_id=None):
  if experiment_id is None:
    parser = mlca.helpers.config.argparser()
    args = parser.parse_args()
    experiment_id = args.experiment_id

  predict_performance_params = PredictPerformanceExperimentList[experiment_id]
  tsp_params = TspExperimentList.get(predict_performance_params.TEST_SYNTHESIZED_PROGRAMS_EXP_NAME)

  with predict_performance_params:
    with tsp_params:
      print(experiment_id, PredictPerformanceParams.current().TEST_SYNTHESIZED_PROGRAMS_EXP_NAME)

      data, curiosity_programs_with_results, \
        curiosity_programs, curiosity_program_inputs, \
        curiosity_data_structures, curiosity_optimizers, \
        reward_combiner_programs, reward_combiner_program_inputs, \
        reward_combiner_data_structures, reward_combiner_optimizers, \
        program_results_data = load_program_data(
          PredictPerformanceParams.current().TEST_SYNTHESIZED_PROGRAMS_EXP_NAME)
      random.seed(3)
      random.shuffle(data)

      X = np.array([program_as_feature_vector_predict_performance(d.curiosity_program)
                    for d in tqdm(data, "Convert programs to feature vectors")
                    if d.stats])
      y = np.array([_get_target_output(d) for d in data if d.stats])

      print("Num datapoints", len(X))

      num_train = int(len(y) * .8)
      train_X, train_y = X[:num_train], y[:num_train]
      test_X, test_y = X[num_train:], y[num_train:]

      if PredictPerformanceParams.current().TASK == "TEST_SINGLE":
        regr = get_predict_performance_regressor()
        regr.fit(train_X, train_y)

        if PredictPerformanceParams.current().MODEL == "GP":
          train_y_pred, train_y_pred_std = regr.predict(train_X, return_std=True)
          test_y_pred, test_y_pred_std = regr.predict(test_X, return_std=True)
        else:
          train_y_pred = regr.predict(train_X)
          test_y_pred = regr.predict(test_X)

        train_r2 = r2_score(train_y, train_y_pred)
        test_r2 = r2_score(test_y, test_y_pred)

        folder = "pickles/predict_performance_data"
        if not os.path.exists(folder):
          os.mkdir(folder)
        with open(f"{folder}/{experiment_id}.txt", "w+") as f:
          def w(*x):
            s = " ".join([str(x) for x in x])
            print(s)
            f.write(s)
            f.write("\n")
          w("------------------")
          w(experiment_id)
          w("------------------")
          w("Num datapoints", len(data))
          # w("Length of feature vector", len(FEATURE_VECTOR_CLASSES))
          # w("Coefficients: \n", regr.coef_)
          w(f"train_r2: {train_r2}")
          w(f"test_r2: {test_r2}")
          w("Mean squared error: %.2f"
            % mean_squared_error(train_y, train_y_pred))
          w("Mean squared error: %.2f"
            % mean_squared_error(test_y, test_y_pred))
          w("Variance score: %.2f" % r2_score(test_y, test_y_pred))
          
          if PredictPerformanceParams.current().MODEL == "GP":
            train_probs = mlca.helpers.probability.pdf(
              train_y,
              train_y_pred,
              train_y_pred_std
            )
            train_bayesian_nll = sum(
              np.minimum(-np.log(train_probs), 10)
            )
            w(f"train bayesian NLL: {train_bayesian_nll}")
            
            test_probs = mlca.helpers.probability.pdf(
              test_y,
              test_y_pred,
              test_y_pred_std
            )
            test_bayesian_nll = sum(
              np.minimum(-np.log(test_probs), 10)
            )
            w(f"test bayesian NLL: {test_bayesian_nll}")

          # if PredictPerformanceParams.current().MODEL == "Linear":
          #   w("Top features")
          #   for coef, classes in sorted(zip(regr.coef_, FEATURE_VECTOR_CLASSES), key=lambda x: x[0], reverse=True)[:10]:
          #     w(coef, classes)

          #   w("Bottom features")
          #   for coef, classes in sorted(zip(regr.coef_, FEATURE_VECTOR_CLASSES), key=lambda x: x[0])[:10]:
          #     w(coef, classes)

        print(regr)

        plt.scatter(train_y, train_y_pred, color="blue", s=.5)
        plt.legend()
        plt.title("Performance Regressor on the Training Set")
        plt.ylabel("Predicted Performance")
        plt.xlabel("Actual Performance")
        plt.plot( [30,500],[30,500], color="black")
        plt.xlim(30, 500)
        # plt.show()
        plt.savefig(f"pickles/predict_performance_data/{experiment_id}_train.jpg")
        plt.clf()

        plt.scatter(test_y, test_y_pred, color="green", s=.5)
        plt.legend()
        plt.title("Performance Regressor on the Test Set")
        plt.ylabel("Predicted Performance")
        plt.xlabel("Actual Performance")
        plt.plot( [30,500],[30,500], color="black")
        plt.xlim(30, 500)
        # plt.show()
        plt.savefig(f"pickles/predict_performance_data/{experiment_id}_test.jpg")
        plt.clf()

      elif PredictPerformanceParams.current().TASK == "TRAINING_POINTS_PLOT":
        train_r2s = []
        test_r2s = []
        training_points = []
        for data_bucket in range(1, 100):
          data_percent = data_bucket / 100
          num_points = int((data_percent) * len(train_X))

          train_X_limited = train_X[:num_points, :]
          train_y_limited = train_y[:num_points]

          regr = get_predict_performance_regressor()
          regr.fit(train_X_limited, train_y_limited)

          train_y_pred = regr.predict(train_X_limited)
          train_r2 = r2_score(train_y_limited, train_y_pred)

          test_y_pred = regr.predict(test_X)
          test_r2 = r2_score(test_y, test_y_pred)

          train_r2s.append(train_r2)
          test_r2s.append(test_r2)
          training_points.append(len(train_X_limited))

        plt.plot(training_points, train_r2s, color="blue", label="train")
        plt.plot(training_points, np.maximum(0,np.array(test_r2s)), color="green", label="test")
        plt.title("Performance Prediction r^2 vs # Training Points")
        plt.xlabel("# Training Points")
        plt.ylabel("r^2")
        plt.savefig(f"scripts/predict_performance_data/{experiment_id}.jpg")
        plt.show()
        plt.clf()

        plt.plot(training_points, np.log(train_r2s), color="blue", label="train")
        plt.plot(training_points, np.log(test_r2s), color="green", label="test")
        plt.title("Performance Prediction r^2 vs # Training Points")
        plt.xlabel("# Training Points")
        plt.ylabel("r^2")
        plt.savefig(f"scripts/predict_performance_data/{experiment_id}_log.jpg")
        # plt.show()
        plt.clf()
      else:
        quit("Missing task")


if __name__ == "__main__":
  main()
