import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import random
import math
from tqdm import tqdm

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from mlca.program import Program, ProgramWithId
from mlca.diversity.density_peaks import performance_cluster
from mlca.search_program_experiments import SearchExperimentList

from mlca.program import Program
from mlca.scripts.analyze_synthesized_programs import load_program_data, load_curiosity_program_data, ProgramData
from mlca.helpers.plotting import scatter_histogram_histogram
from mlca.test_synthesized_programs_experiments import TspParams, TspExperimentList
from mlca.predict_performance import program_as_feature_vector_diversity
import mlca.helpers.config

def _get_target_output(d: ProgramData):
  return d.stats["mean_performance"]

def main():
  # exp_name_1 = "2-13-15x15-ppo-5-trials"
  # exp_name_2 = "2-14-15x15-ppo-10-trials"
  # exp_name_1 = "2-15-15x15-ppo-20-trials"
  # exp_name_2 = "2-16-15x15-ppo-5-trials-share-curiosity"
  # exp_name = "2-17-15x15-ppo-10-trials-share-curiosity"

  # exp_name_1 = "2-24-15x15-ppo-10-rollouts"
  # exp_name_2 = "2-27-15x15-ppo-10-rollouts-500-steps"

#   exp_name_1 = "2-25-15x15-ppo-5-rollouts--version_b"
#   exp_name_2 = "2-26-15x15-ppo-1-rollout--version_b"
  # exp_name_1 = "2-28-15x15-ppo-5-rollouts-500-steps--version_b_small"
  # exp_name_2 = "2-28-15x15-ppo-5-rollouts-500-steps--version_b_small"

  # exp_name_1 = "2-28-15x15-ppo-5-rollouts-500-steps--version_b_small"
  # exp_name_2 = "2-28-15x15-ppo-5-rollouts-500-steps--version_b_small"
  # exp_name_2 = "2-29-15x15-ppo-1-rollout-500-steps"
  # exp_name_2 = "2-30-15x15-ppo-1-rollout-500-steps-10-episodes-per-rollout"
  # exp_name_1 = "2-31-15x15-ppo-1-rollout-500-steps-10-episodes-per-rollout-four-rooms"
  # exp_name_1 = "2-32-15x15-ppo-1-rollout-500-steps-four-rooms"

  # Same as 2-28-15x15-ppo-5-rollouts-500-steps--version_b_small
  # exp_name_2 = "2-32-15x15-ppo-1-rollout-500-steps-four-rooms"

  # exp_name_1 = "2-45-15x15-ppo-5-rollouts-500-steps-gcloud-v100" # empty room
  # exp_name_2 = "2-57-15x15-ppo-5-rollouts-500-steps-four-rooms" # four room

  # exp_name_1 = "2-55-15x15-ppo-1-rollout-500-steps-reward-combiner-new-program-list"
  # exp_name_2 = "2-55-15x15-ppo-1-rollout-500-steps-reward-combiner-new-program-list"

  # exp_name_1 = "2-56-15x15-ppo-1-rollout-500-steps-reward-combiner-new-program-list--version_b"
  # exp_name_2 = "2-56-15x15-ppo-1-rollout-500-steps-reward-combiner-new-program-list--version_b"

#   exp_name_1 = "2-45-15x15-ppo-5-rollouts-500-steps-gcloud-v100"  # empty room
#   exp_name_2 = "2-59-15x15-ppo-5-rollouts-500-steps-four-rooms"

#   exp_name_1 = "2-60-15x15-ppo-5-rollouts-500-steps-lunar-lander"
#   exp_name_2 = "2-61-15x15-ppo-5-rollouts-500-steps-lunar-lander-shared-curiosity"

#   exp_name_2 = "2-73-15x15-ppo-5-rollouts-500-steps-lunar-lander-batched-ppo"

#   exp_name_1 = "2-62-15x15-ppo-5-rollouts-500-steps-lunar-lander"
#   exp_name_2 = "2-66-15x15-ppo-5-rollouts-500-steps-gridworld"

#   exp_name_1 = "2-60-15x15-ppo-5-rollouts-500-steps-lunar-lander"
#   exp_name_2 = "2-64-ppo-5-rollouts-seaquest-ram"

#   exp_name_1 = "2-60-15x15-ppo-5-rollouts-500-steps-lunar-lander"
#   exp_name_2  = "2-73-15x15-ppo-5-rollouts-500-steps-lunar-lander-batched-ppo"

#   exp_name_1 = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"
#   exp_name_1 = "2-98_acrobot_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"
#   exp_name_2 = "2-100_lunar_lander_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"
#   exp_name_2 = "2-101_acrobot_distribution_new_atari-ppo-real-batched-shared_7500-steps_2-trials-yes-share-yes-batch-1_steps_curiosity"

  # exp_name_1 = "2-103_ant_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"
  # exp_name_2 = "2-103_hopper_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"

  # exp_name_1 = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"
  # exp_name_2 = "2-96_regression-test

  exp_name_1 = "2-96_15x15_new-ppo-real-batched-shared_1000-steps_5-trials-yes-share-yes-batch-1_steps_curiosity"
  exp_name_2 = "2-111_lunar-lander_diversity_combined"

  params_1 = TspExperimentList.get(exp_name_1)
  params_2 = TspExperimentList.get(exp_name_2)

  assert params_1.CURIOSITY_PROGRAMS_NAME == params_2.CURIOSITY_PROGRAMS_NAME
  assert params_1.EXPERIMENT_TYPE == params_2.EXPERIMENT_TYPE

  print(params_1)

  data_1, _, _, _, _, _, _, _, _, _, _\
    = load_program_data(exp_name_1)
  data_2, _, _, _, _, _, _, _, _, _, _\
    = load_program_data(exp_name_2)

  print("Data len", len(data_1), len(data_2))

  # Select best programs to run in mujoco
  exp_name_3 = "2-112_acrobot_diversity_combined"
  params_3 = TspExperimentList.get(exp_name_3)
  data_3, _, _, _, _, _, _, _, _, _, _\
    = load_program_data(exp_name_3)
  meta_select_programs(
    params_1,
    data_1, data_2, data_3
  )
  quit()

  # comparisons = [ compare_programs(d1, d2) for d1, d2 in zip(data_1, data_2) ]

  # data_2_by_index = {p.index: p for p in data_2}
  # d = [
  #   (p1, data_2_by_index[p1.index]) 
  #   for p1 in data_1 
  #   if p1.index in data_2_by_index and p1.stats and data_2_by_index[p1.index].stats]

  if params_1.EXPERIMENT_TYPE == TspParams.ExperimentType.CURIOSITY_SEARCH:
    data_2_by_program_id = {p.curiosity_program.program_id: p for p in data_2}
    d = [
        (p1, data_2_by_program_id[p1.curiosity_program.program_id])
        for p1 in data_1
        if p1.curiosity_program.program_id in data_2_by_program_id
        and p1.stats and data_2_by_program_id[p1.curiosity_program.program_id].stats]

    for p in d:
      assert p[0].curiosity_program.program_id == p[1].curiosity_program.program_id, (
          p[0].curiosity_program.program_id, p[1].curiosity_program.program_id)
  elif params_1.EXPERIMENT_TYPE == TspParams.ExperimentType.REWARD_COMBINER_SEARCH:
    data_2_by_program_id = {
        p.reward_combiner_program.program_id: p for p in data_2}
    d = [
        (p1, data_2_by_program_id[p1.reward_combiner_program.program_id])
        for p1 in data_1
        if p1.reward_combiner_program.program_id in data_2_by_program_id
        and p1.stats and data_2_by_program_id[p1.reward_combiner_program.program_id].stats]


  # d = [p for p in zip(data_1, data_2) if p[0].stats and p[1].stats]

  # for p1 in data_1:
  #   print(p1.index)


  # for p in d:
  #   print([p[0].stats["mean_performance"], p[1].stats["mean_performance"]])
  # random.shuffle(d)
  data_1 = [p[0] for p in d]
  data_2 = [p[1] for p in d]

  best = [p for p in d if p[0].stats["mean_performance"] > 550 and p[1].stats["mean_performance"] > -200]
  print("# best", len(best))
  for i, d in enumerate(best):
    d[0].curiosity_program.visualize_as_graph(i)

  means_1 = [_get_target_output(d) for d in data_1]
  means_2 = [_get_target_output(d) for d in data_2]
  print(len(data_1))
  print(len(data_2))

  X = np.array(means_1).reshape(-1, 1)
  y = np.array(means_2)

  print(X.shape, y.shape)

  print("Raw r^2 (no prediction): %.2f"
        % r2_score(means_1, means_2))

  print("Num datapoints", len(data_1), len(data_2))

  num_train = int(len(y) * .8)
  train_X, train_y = X[:num_train], y[:num_train]
  test_X, test_y = X[num_train:], y[num_train:]

  regr = linear_model.LinearRegression()
  regr.fit(train_X, train_y)

  print("train_X", train_X.shape)

  train_y_pred = regr.predict(train_X)
  train_r2 = regr.score(train_X, train_y)

  test_y_pred = regr.predict(test_X)
  test_r2 = regr.score(test_X, test_y)

  # print('Coefficients: \n', regr.coef_)
  print(f"train_r2: {train_r2}")
  print(f"test_r2: {test_r2}")
  print("Mean squared error: %.2f"
        % mean_squared_error(train_y, train_y_pred))
  print("Mean squared error: %.2f"
        % mean_squared_error(test_y, test_y_pred))
  print('Variance score: %.2f' % r2_score(test_y, test_y_pred))

  print(len([_get_target_output(d) for d in data_1]))
  a = np.array([_get_target_output(d) for d in data_1])
  b = np.array([_get_target_output(d) for d in data_2])
  a_std = np.array([d.stats["mean_performance_std"] for d in data_1])
  b_std = np.array([d.stats["mean_performance_std"] for d in data_2])
  plt.errorbar(
    a, 
    b,
    a_std,
    b_std,
    color="blue", label="train") # , s=.5)
  plt.xlabel(exp_name_1)
  plt.ylabel(exp_name_2)
  plt.show()

  scatter_histogram_histogram(
      [_get_target_output(d) for d in data_1],
      [_get_target_output(d) for d in data_2],
      binwidth=5, 
      xlabel=exp_name_1, 
      ylabel=exp_name_2
  )

#   plt.scatter(
#       train_y,
#       train_y_pred,
#       color="blue", label="train", s=.5)
#   plt.xlabel(exp_name_1)
#   plt.ylabel("PREDICT " + exp_name_2)
#   plt.show()

def meta_select_programs(
  params_1,
  data_1, data_2, data_3):
  tested_program_ids =  [42016, 42018, 42022, 38031, 42019, 42015, 7551, 42023, 42012, 42011, 38029, 40993, 42010, 8409, 8211, 21414, 32631, 10724, 28869, 10695, 28728, 8667, 43317, 33302, 41214, 42096, 42092, 13128, 47764, 39339, 51859, 38126, 51820, 32103, 15565, 8599, 51120, 51817, 15546, 51856, 25815, 41894, 24440, 10709, 48896, 12638, 35298, 15566, 10232, 10542, 51864, 29274, 12632, 15375, 35458, 30615, 8571, 42153, 35376, 14134, 15488, 47207, 25981, 22520, 31547, 35411, 51108, 15709, 48103, 3099, 140, 38825, 37015, 31327, 7935, 28572, 461, 420, 50491, 21843, 20473, 40139, 34678, 15710, 15735, 59, 45620, 35785, 36989, 51100, 35435, 15977, 32937, 31490, 9999, 45644, 10181, 41292, 51080, 42254, 15823, 401, 4488, 45812, 737, 7924, 42199, 35174, 15824, 35438, 43353, 33896, 50838, 14368, 29050, 19768, 10229, 41241, 19472, 14288, 15860, 333, 10166, 42196, 4094, 15528, 41655, 7409, 49402, 29269, 10022, 28870, 48665, 51273, 37485, 39662, 20378, 10413, 50919, 2272, 9187, 24869, 50255, 44987, 6303, 37929, 19523, 15405, 64, 10000, 35247, 51839, 13595, 15910, 1933, 42779, 15708, 15895, 20554, 6304, 31771, 34015, 32332, 16081, 41553, 707, 45304, 34954, 41446, 47382, 27866, 37977, 10450, 9546, 27367, 49437, 35444, 43194, 46679, 38, 14666, 47202, 41987, 9600, 9450, 43839, 34531, 50037, 1392, 51502, 9557, 32972, 19091, 28873, 16340, 27905, 33775, 15559, 41464, 267, 34728, 20959, 4953, 30646, 48473, 15136, 16078, 16322, 42916, 10167, 15439, 28888, 15945, 35172, 15732, 27720, 27648, 9723, 24661, 37104, 19920, 7560, 30997, 14701, 25591, 38495, 15221, 19501, 14463, 36595, 51739, 10044, 15863, 30296, 16865, 41971, 22691, 41134, 42484, 19767, 28243, 44301, 14491, 38268, 41376, 47910, 38332, 44790, 9547, 15825, 32670, 27909, 33840, 35721, 42964, 531, 10376, 16109, 31133, 1946, 16079, 26866, 35176, 42354, 44111, 14729, 17197, 44553, 38365, 15435, 35271, 37340, 44733, 48435, 15316, 40974, 41794, 9317, 33825, 47498, 48057, 15731, 716, 44077, 44145, 10921, 14492, 51628, 44795, 646, 16578, 32336, 9132, 36793, 9735, 16859, 51149, 44169, 50564, 9944, 43578, 10559, 35177, 31757, 41543, 28935, 34948, 33862, 48631, 14119, 16912, 45238, 44581, 15885, 34949, 51135, 16599, 15581, 33707, 34974, 35685, 46249, 7439, 41641, 47327, 1917, 11205, 50250, 2438, 10068, 38187, 8747, 9117, 46068, 33886, 15705, 16146, 37537, 38503, 33552, 15867, 31130, 37197, 27431, 29056, 38338, 51358, 6108, 36586, 15911, 15949, 46640, 43224, 32621, 10696, 43241, 51208, 37028, 41822, 34909, 37334, 139, 38327, 43988, 45337, 10860, 28316, 15946, 32534, 42183, 43867, 10337, 17021, 17065, 51103, 15451, 16711, 36306, 37106, 44212, 8895, 50513, 1467, 12050, 25600, 38515, 41117, 15947, 44180, 20856, 992, 28743, 462, 8090, 8637, 15663, 11617, 37175, 38405, 33497, 49904, 50699, 15951, 16298, 9687, 11202, 35976, 51689, 17375, 20860, 43757, 49307, 11164, 27854, 6564, 14641, 15436, 33983, 41006, 41920, 29901, 41629, 310, 15666, 43881, 185, 41389, 41731, 16466, 22570, 37399, 38219, 51803, 37570, 47487, 48253, 31852, 11038, 36729, 19, 15315, 29344, 35903, 10513, 11665, 14938, 19706, 38182, 51518, 2131, 28067, 9480, 163, 15313, 28717, 20334, 81, 1045, 38606, 42132, 44248, 44673, 45483, 16698, 31329, 32717, 40985, 42996, 9065, 16061, 14073, 51061, 31620, 9406, 14635, 28304, 43780, 43874, 411, 19502, 37482, 43886, 656, 16509, 43511, 14703, 15320, 16390, 22852, 37478, 50841, 14185, 16601, 25573, 34947, 3761, 11970, 15712, 16490, 19761, 42687, 9865, 10556, 16628, 16880, 43504, 26235, 32962, 35816, 43687, 7498, 15564, 21, 14856, 1006, 15840, 51749, 15966, 43238, 25046, 20052, 44044, 309, 43230, 9202, 16960, 42840, 50351, 14153, 17196, 34912, 43364, 44160, 16259, 29332, 10418, 35446, 616, 18127, 28893, 36841, 9281, 9495, 27779, 43229, 6855, 29732, 7484, 43498, 43501, 44122, 44986, 239, 44080, 48829, 19399, 38020, 51282, 51426, 37394, 51092, 189, 16383, 17387, 27661, 32445, 15512, 15868, 51306, 1417, 15817, 15887, 35487, 16657, 16806, 50417, 27606, 10266, 42307, 44063, 18911, 1936, 40251, 40749, 9050, 11417, 29371, 47997, 9038, 43372, 44168, 15886, 11048, 39721, 43678, 51365, 15326, 15524, 37580, 63, 2379, 23648, 596, 17559, 19859, 29296, 15358, 31807, 43596, 47619, 45045, 2725, 10007, 21790, 35422, 33978, 31385, 1403, 7238, 9253, 28852, 37560, 38416, 31167, 41747, 47881, 50845, 51658, 9473, 27234, 29970, 1800, 29934, 35682, 16408, 17292, 49474, 422, 5336, 35223, 43147, 47962, 17328, 36467, 44117, 51118, 9725, 14429, 17535, 25359, 33129, 37587, 1646, 50978, 594, 14484, 11610, 15289, 25292, 10691, 11392, 33220, 51741, 2690, 10138, 49681, 8882, 27457, 38475, 12355, 16640, 38068, 45849, 468, 27267, 44944, 54, 8678, 16284, 43925, 2092, 16882, 69, 10582, 44347, 9284, 37593, 49040, 13370, 26236, 26641, 8886, 9236, 383, 30593, 32648, 43540, 9072, 44059, 49770, 14363, 44515, 44544, 26078, 44780, 18682, 187, 11891, 45067, 16069, 32119, 50983, 51037, 15981, 24261, 4821, 15440, 16967, 19662, 22606, 45146, 51770, 37683, 6655, 10280, 44368, 44779, 45383, 51133, 51281, 15826, 33837, 42189, 51026, 51191, 332, 17771, 25706, 35491, 36464, 43508, 11673, 15753, 10340, 36438, 38463, 41803, 2074, 15845, 20830, 44148, 491, 11662, 12051, 15909, 17529, 1904, 2902, 8662, 11497, 29391, 41016, 15870, 34955, 46671, 49231, 1042, 46, 4064, 9382, 23037, 42748, 44144, 51823, 28336, 34366, 37948, 44476, 45101, 364, 2617, 15765, 44861, 44874, 382, 31198, 43346, 44428, 42, 9283, 19580, 38498, 8610, 48034, 293, 323, 2251, 7561, 43893, 11814, 9539, 16190, 14311, 15880, 16068, 17610, 18128, 24021, 9110, 29316, 44723, 3467, 51646, 16606, 21994, 25036, 35620, 49036, 15616, 46996, 41520, 45495, 49, 22079, 6582, 37603, 49033, 2057, 44159, 51073, 14391, 49824, 14922, 34571, 41003, 10677, 30321, 120, 2483, 18562, 35615, 44126, 14040, 41456, 32701, 41099, 43939, 43969, 44929, 642, 30447, 11674, 15291, 38485, 1639, 10221, 22409, 33624, 37546, 34353, 47290, 10193, 14802, 41743, 50596, 16083, 16769, 24192, 44410, 31433, 40655, 2254, 16776, 511, 11924, 14897, 15529, 32168, 16763, 454, 6, 13, 339, 8883, 39365, 44995, 9101, 9289, 14202, 28290, 42226, 50929, 1986, 8825, 10259, 16950, 44788, 47942, 4042, 10813, 26805, 51735, 7391, 37315, 44474, 352, 14385, 36856, 47246, 51576, 15936, 41891, 43214, 44205, 44087, 47314, 30, 482, 16981, 20261, 36945, 41308, 47645, 51218, 745, 930, 40930, 3568, 9271, 19943, 37536, 51243, 117, 18381, 21088, 33231, 14389, 18388, 37914, 43515, 17, 11940, 34435, 15677, 18946, 44889, 42321, 43781, 49661, 15314, 17810, 31458, 36175, 45014, 48097, 11820, 31446, 49838, 8780, 42792, 50961, 934, 50, 34472, 44970, 9288, 43914, 44923, 11831, 13945, 31154, 35696, 43213, 44385, 51292, 13886, 30674, 44181, 70, 20550, 38590, 44935, 9433, 35587, 16808, 26731, 51657, 2274, 41236, 2344, 16435, 43590, 44506, 31017, 44507, 18336, 47330, 51194, 15737, 15882, 18916, 16896, 1, 10024, 47724, 51235, 51516, 31625, 51142, 34467, 49760, 34963, 48818, 23, 9827, 9278, 34432, 43216, 43906, 44166, 44938, 45000, 814, 10437, 32926, 15931, 102, 27594, 2067, 43226, 51186, 2250, 9269, 14998, 15521, 18941, 44380, 44556, 45268, 50222, 51764, 2240, 51508, 3466, 15537, 50522, 47, 2289, 10510, 10720, 11254, 15607, 46048, 16907, 31152, 43921, 51, 9199, 10197, 26629, 15558, 48629, 23337, 29506, 480, 4073, 9088, 15357, 16469, 51210, 11062, 16087, 41626, 16800, 17583, 19708, 30894, 37475, 607, 10272, 44348, 51290, 1005, 9085, 44892, 9156, 13975, 15800, 9734, 255, 10551, 14376, 44165, 51711, 51858, 682, 51505, 126, 43270, 0, 10062, 44496, 45022, 18414, 354, 30004, 9099, 43558, 15561, 42694, 51344, 9449, 15256, 45043, 51345, 24571, 4052, 23861, 41810, 45855, 15245, 36644, 38406, 13325, 38449, 44353, 9205, 11755, 41378, 43205, 44997, 11243, 31020, 44481, 46992, 51349, 2257, 15502, 42326, 44495, 47545, 55, 15769, 24003, 8695, 10716, 33885, 43622, 44523, 51346, 51555, 8, 894, 9169, 320, 13831, 26667, 11976, 16452, 8647, 50129, 38562, 16397, 28759, 45827, 220, 11547, 11712, 43472, 44910, 46245, 18762, 43392, 50788, 43698, 44917, 10945, 13961, 17408, 43697, 28823, 33575, 51865, 16571, 44095, 51850, 509, 1138, 15633, 15669, 43722, 50997, 51352, 51733, 16242, 208, 2281, 15486, 43290, 53, 327, 831, 6066, 8939, 15881, 44547, 5, 7, 65, 41277, 15796, 19622, 37980, 44239, 13814, 42066, 804, 43625, 44533, 14282, 15784, 14107, 2839, 44072, 6720, 48839, 51789, 759, 11679, 16411, 17366, 19033, 37292, 51648, 4056, 16353, 16799, 43879, 44494, 89, 10611, 30233, 34170, 51525, 16863, 14076, 16971, 37030, 2427, 44909, 51342, 43394, 43866, 44863, 14916, 17013, 10614, 14383, 33083, 93, 112, 15242, 16192, 30263, 43494, 50227, 51425, 27740, 49777, 32746, 35483, 44134, 44157, 44156, 125, 15681, 38237, 41426, 51398, 11507, 16097, 45113, 47520, 51355, 9843, 168, 3654, 1643, 14276, 16761, 51421, 8703, 8657, 37528, 16232, 38714, 43847, 26637, 15738, 14302, 22915, 25705, 29923, 30003, 1847, 44011, 46688, 14404, 27770, 35059, 43462, 2280, 15694, 16756, 32280, 51117, 14399, 14441, 15879, 16076, 14409, 26432, 26493, 31248, 15850, 18907, 44543, 36005, 11174, 16545, 43918, 44919, 48853, 8629, 15499, 1983, 29988, 37414, 11827, 31675, 50946, 2354, 44491, 50371, 9479, 25750, 44877, 2103, 28565, 43507, 41661, 1931, 898, 48394, 17873, 27301, 42310, 44478, 37347, 15491, 16721, 27455, 2358, 1011, 2298, 14275, 16816, 8697, 9243, 10060, 15369, 15736, 106, 484, 715, 16678, 43755, 47567, 7738, 37613, 44513, 2, 10268, 2489, 28872, 12009, 11824, 13847, 601, 35015, 11621, 51351, 15650, 16406, 36610, 26902, 36149, 10156, 16647, 51196, 11639, 48644, 414, 10271, 15771, 43251, 15980, 44992, 5351, 11851, 10560, 11227, 16496, 2063, 16948, 9052, 11869, 44535, 533, 44214, 21464, 1578, 15627, 23719, 107, 9736, 45122, 9111, 16622, 51325, 14397, 51233, 11550, 22090, 43617, 15659, 681, 3212, 40585, 44941, 1653, 16758, 9143, 15746, 23957, 35979, 51244, 9116, 30326, 9091, 40979, 9144, 10191, 26031, 33370, 38469, 51660]

  def score_programs(p_data):
    for p in tested_program_ids:
      print(p, p_data[p].stats)
    p_by_score = list(sorted(
      tested_program_ids, key = lambda p: \
        p_data[p].stats["mean_performance"] 
        if p_data[p].stats else -math.inf
    ))
    return {
      p: i / len(p_by_score)
      for i, p in enumerate(p_by_score)
    }

  id_to_data_1 = {d.curiosity_program.program_id: d for d in data_1}
  id_to_data_2 = {d.curiosity_program.program_id: d for d in data_2}
  id_to_data_3 = {d.curiosity_program.program_id: d for d in data_3}

  id_to_score_1 = score_programs(id_to_data_1)
  id_to_score_2 = score_programs(id_to_data_2)
  id_to_score_3 = score_programs(id_to_data_3)

  program_scores = {
    p: id_to_score_1[p] + id_to_score_2[p] + id_to_score_3[p]   
    for p in tested_program_ids
  }

  tested_program_id_scores = [
    id_to_score_1[p] + id_to_score_2[p] + id_to_score_3[p]   
    for p in tested_program_ids
  ]
  simulator_params = SearchExperimentList["ss-knn_10-fv_1_pairs-early_termination-diversity-knn"]
  with simulator_params:
    with params_1: 
      tested_program_features = [
        program_as_feature_vector_diversity(
          id_to_data_1[p].curiosity_program
        ).astype(np.float)
        for p in tqdm(tested_program_ids, "Make program features")
       ]

      # DELTA_THRESHOLD = 3.5
      # PERFORMANCE_THRESHOLD  = 400

      DELTA_THRESHOLD = 7.27
      PERFORMANCE_THRESHOLD = 2
      qualifying_point_indices, deltas, parent_pointers = performance_cluster(
        tested_program_ids, tested_program_id_scores, tested_program_features, 
        DELTA_THRESHOLD, PERFORMANCE_THRESHOLD, visualize=True)

      print("qualifying_point_indices", qualifying_point_indices)
      for i, p in enumerate(qualifying_point_indices):
        print("--")
        print(
          p, 
          program_scores[p], 
          _get_target_output(id_to_data_1[p]),
          _get_target_output(id_to_data_2[p]),
          _get_target_output(id_to_data_3[p]),
        )
        id_to_data_1[p].curiosity_program.visualize_as_graph(i)

      quit()

  programs_by_score = list(sorted(
    tested_program_ids, key = lambda p: program_scores[p]
  ))
  print(program_scores)
  print(programs_by_score)
  
  n = 16
  top_n = programs_by_score[-n:]
  print(top_n)
  for i, p in enumerate(reversed(top_n)):
    print("--")
    print(
      p, 
      program_scores[p], 
      _get_target_output(id_to_data_1[p]),
      _get_target_output(id_to_data_2[p]),
      _get_target_output(id_to_data_3[p]),
    )
    id_to_data_1[p].curiosity_program.visualize_as_graph(i)



if __name__ == "__main__":
    main()
