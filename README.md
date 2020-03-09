# Meta-Learning Curiosity Algorithms
"Meta-Learning Curiosity Algorithms" by [Ferran Alet](http://alet-etal.com/)\*, [Martin Schneider](https://github.com/mfranzs)\*, [Tomas Lozano-Perez](https://people.csail.mit.edu/tlp/), and [Leslie Kaelbling](https://people.csail.mit.edu/lpk/). 2019. “Meta-Learning Curiosity Algorithms.” In Meta-Learning and Reinforcment Learning Workshops @NeurIPS. 

See the paper [here](https://openreview.net/pdf?id=BygdyxHFDS). 

## Overview of Running an Experiment
1. Specify your operations in operations.py.
2. Specify a list of operations to use in operations_list.py.
3. Run program_synthesis.py to synthesize programs with your list of operations.
4. Specify an experperiment in test_synthesized_programs_experiments.py.
5. Run test_synthesized_programs.py to search over your program space.
6. Use scripts/analyze_synthesized_programs.py to analyze your results.

## Code Overview
**datastructures.py**: The datastructures manipulated by program operations.\
**executor.py**: Executes a Program object.\
**find_duplicate_programs.py**: Takes a list of programs and finds / prunes duplicates by testing each program on a fake environment and looking at the output signature.\
**gridworld_environments.py**: Our gridworld environments.\
**internal_rewards.py**: The module that runs intrinsic curiosity programs and reward combiner programs.\
**operations_list.py**: A configuration file that specifies the operations that can appear in different program classes\
operations.py: The operations that are composed to create a program.\
**predict_performance.py**: The regressor that predicts program performance from its \
**predict_performance_experiments.py**: A configuration file for experimenting with performance regressors.\
**program.py**: The core abstraction of a program, represented by a DAG of operations**.\
**program_synthesis.py**: The search module that synthesizes programs.\
**program_types.py**: The types that operations in our language can output.\
**run_agent.py**: The module that runs an agent in an environment.\
**search_programs.py**: The module that searches over a program space, given a list of programs, an environment, and a program selection metric.\
**search_programs_experiments.py**: A configuration file for simulating program searches.\
**simulate_search.py**: A module that simulates searching through programs.\
**test_synthesized_programs.py**: The module that takes a set of synthesized programs and initiates a search over them.\
**test_synthesized_programs_experiments.py**: The configuration file for testing / searching over programs.\
