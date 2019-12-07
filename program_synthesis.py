"""
Synthesize a list of programs, given a list of operations defined in 
operations.py.
"""

import mlca.program_types as program_types
from mlca.program import Program
from mlca.find_duplicate_programs import get_program_signature
from mlca.gridworld_environments import EmptyEnv
from mlca.test_synthesized_programs_experiments import TspExperimentList
import mlca.operations_list as operations_list
import mlca.operations as operations
import mlca.helpers.config

import itertools
import time
import torch
import pickle
from typing import List, Dict, Set

ProgramNumber = str
ProgramSignature = str

num_searches_that_reached_desired_output: int
duplicates_found: int

FORWARD_STAGE = 1
UPDATE_STAGE = 2

"""
Search to synthesize new programs, using a backtracking search. 
"""
def search(
        inputs, data_structures, optimizers,
        initial_program_list,
        operations_set,
        REQUIRE_UPDATE_PROGRAM,
        MAX_TOTAL_PROGRAM_LENGTH,
        print_valid_programs=False,
        dont_prune_duplicates=False,
        evaluate_to_prune_duplicate_programs=True,
        TARGET_OUTPUT_TYPE=program_types.RealNumber,
        print_filter=None):

    def add_graph_node(operation: operations.Operation, var_type):
        var_type_class = var_type.__class__
        assert var_type_class != program_types.Type, (
            var_type_class.__base__, operation, var_type)
        for t in program_types.type_and_supertypes(var_type_class):
            if t not in variables_by_type:
                variables_by_type[t] = set()

            variables_by_type[t].add(operation)

    def remove_graph_node(operation: operations.Operation, var_type):
        var_type_class = var_type.__class__
        for t in program_types.type_and_supertypes(var_type_class):
            variables_by_type[t].remove(operation)

    def program_uses_all_intermediates(program_construction_stage):
        prgm = current_forward_program if program_construction_stage == FORWARD_STAGE else current_update_program
        for operation_instance in prgm[:-1]:
            if operation_instance.cached_output_type != program_types.Void and operation_instance.num_uses_in_program == 0:
                return False
        return True

    def program_uses_all_required_variables():
        for var in inputs:
            if var.must_be_used and var.num_uses_in_program == 0:
                return False
        return True

    def num_intermediates_used(program_construction_stage):
        prgm = current_forward_program if program_construction_stage == FORWARD_STAGE else current_update_program

        used = 0
        for operation_instance in prgm:
            if operation_instance.cached_output_type != program_types.Void and operation_instance.num_uses_in_program == 0:
                pass
            else:
                used += 1

        return used, len(prgm)

    def program_number_comparator(a, b):
        if len(a) == len(b):
            return a > b
        else:
            return len(a) > len(b)

    def inputs_sorted_for_commutative(operation, input_tuple):
        if type(input_tuple[1]) == type(operation) and \
            input_tuple[1].inputs[0].program_number() > operation.program_number():
            return False
        
        if type(input_tuple[0]) == type(operation) and \
                input_tuple[0].inputs[1].program_number() > operation.program_number():
            return False

        return numbers_are_increasing(i.program_number() for i in input_tuple)
        # # If commutative, reduce duplicates by forcing recursion to go to the right
        # return not input_tuple[0].commutative

    def variable_usage_allowed_in_current_phase(variable: operations.Variable, program_construction_stage):
        if program_construction_stage == UPDATE_STAGE:
            # Don't allow using a data structure in the UPDATE stage if we didn't use it in the FORWARD stage
            return not (variable.is_data_structure and variable.num_uses_in_program == 0)
        else:
            return not variable.only_allowed_in_update

    def can_complete_program_before_length_limit(program_construction_stage):
        n_intermediates_used, program_length = num_intermediates_used(
            program_construction_stage)
        n_intermediates_unused = program_length - n_intermediates_used
        remaining_allowed_operations = MAX_TOTAL_PROGRAM_LENGTH - \
            len(current_forward_program) - len(current_update_program)
        return n_intermediates_unused <= remaining_allowed_operations

    def filter_allowed_variables(variables, program_construction_stage):
        return [i for i in variables 
                if variable_usage_allowed_in_current_phase(i, program_construction_stage) \
                and not (i.can_only_use_once and i.num_uses_in_program > 0)
                and not (i.is_data_structure and not already_using_earlier_data_structures_of_same_type(i))]

    def already_using_earlier_data_structures_of_same_type(data_structure: operations.Variable):
        """If we have two datastructures of the same type, we want to make sure we
        don't accidently generate effectively duplicate programs that just swap which
        datastructure is used. To fight this, we group each datastructure of the same
        type into an ordering and enforce that a datastructure can only be used if all 
        the earlier datastructures in the ordering are also used."""
        t = type(data_structure.var_type)
        for var in data_structures_by_type[t]:
            if var == data_structure:
                return True # We reached our datastructure in the scan, so we're already using everything before
            elif var.num_uses_in_program == 0:
                return False
            else: 
                pass # Continue scanning until reach our datastructure
        raise RuntimeError(f"Data structure not registered. {data_structure} {t} {data_structures_by_type[t]}")

    def operation_allowed_in_current_phase(operation, program_construction_stage):
        return not (program_construction_stage == UPDATE_STAGE and operation.forbiden_in_update_phase)
    
    def add_current_program_to_results_list():
        add_program_to_results_list(Program(
            current_forward_program.copy(),
            current_update_program.copy(),
            inputs, 
            data_structures,
            optimizers,
            len(programs),
            None
            ))

    def add_program_to_results_list(program: Program):
        if evaluate_to_prune_duplicate_programs:
            program_signature = get_program_signature(program, test_env)

            if program_signature in seen_program_signatures:
                global duplicates_found
                duplicates_found += 1
            else:                
                seen_program_signatures[program_signature] = [program] 
                programs.append(program)
        else:
            programs.append(program)

            if len(programs) % 10000 == 0:
                print(
                    f"Found {len(programs)} programs " +
                    f"in {time.time() - start_time} seconds. ")

            if print_valid_programs:
                print(program)

    programs: List[Program] = []
    
    seen_program_signatures: Dict[ProgramSignature, Program] = {}
    exp_name = "2-80_30x30_new-ppo-real-batched-shared_2500-steps_5-trials" # Need to get a params dict, so use an arbitrary experiment
    params = TspExperimentList[exp_name]

    with params:
        test_env = EmptyEnv()

        # Backtracked data in the recursive search:
        current_forward_program: List[operations.Operation] = []
        current_update_program: List[operations.Operation] = []

        variables: List[operations.Operation] = []
        variables_by_type: Dict[program_types.ProgramType, Set[operations.Operation]] = {}
        data_structures_by_type: Dict[program_types.ProgramType, List[operations.Operation]] = {}

        existing_program_numbers: Set[ProgramNumber] = set()

        start_time = time.time()

        global num_searches_that_reached_desired_output
        num_searches_that_reached_desired_output = 0
        global duplicates_found
        duplicates_found = 0

        def recursive_search(min_program_number: ProgramNumber, program_construction_stage=FORWARD_STAGE, recursion_depth=0):
            # If we can't finish in time, terminate
            if not can_complete_program_before_length_limit(program_construction_stage):
                # TODO: This heuristic has some false terminations.
                # If an operation is able to use two or more intermediates at a time, we could use up all intermediates even if this heuristic says we can't.
                # Is this worth fixing?
                return

            if len(current_forward_program) + len(current_update_program) >= MAX_TOTAL_PROGRAM_LENGTH:
                return

            # Do the search
            for operation in operations_set.OPERATIONS:
                if not operation_allowed_in_current_phase(operation, program_construction_stage):
                    continue

                # For each of this operation's inputs, find a list of variables or intermediate
                # operations with matching types.
                variables_for_input = [
                    filter_allowed_variables(variables_by_type.get(
                        input_type, []), program_construction_stage)
                    for input_type in
                    operation.input_program_types
                ]

                input_options = list(itertools.product(*variables_for_input))
                for input_tuple in input_options:
                    if (not operation.commutative or inputs_sorted_for_commutative(operation, input_tuple)) \
                            or dont_prune_duplicates:

                        output_type = operation.output_type_fn(
                            [i.cached_output_type for i in input_tuple])
                        if output_type != operations.INVALID_INPUTS and operation.inputs_allowed(input_tuple):
                            assert issubclass(output_type.__class__, program_types.Type), (output_type, operation, input_tuple, [
                                i.cached_output_type for i in input_tuple])

                            operation_instance = operation(
                                *input_tuple, cached_output_type=output_type)
                            operation_program_number = operation_instance.program_number()

                            if (program_number_comparator(operation_program_number, min_program_number)
                                and operation_program_number not in existing_program_numbers) \
                            or dont_prune_duplicates:

                                # Extend program & available variables
                                add_graph_node(operation_instance, output_type)

                                for i in input_tuple:
                                    i.num_uses_in_program += 1

                                new_program_construction_stage = program_construction_stage
                                new_min_program_number = operation_program_number
                                existing_program_numbers.add(
                                    operation_program_number)

                                if program_construction_stage == FORWARD_STAGE:
                                    current_forward_program.append(
                                        operation_instance)

                                    if program_types.equal_or_supertype(operation_instance.cached_output_type.__class__, TARGET_OUTPUT_TYPE) \
                                            and program_uses_all_intermediates(program_construction_stage) \
                                            and program_uses_all_required_variables():

                                        new_program_construction_stage = UPDATE_STAGE
                                        new_min_program_number = ""
                                        global num_searches_that_reached_desired_output
                                        num_searches_that_reached_desired_output += 1

                                        already_mutated_value = any(o.mutates_datastructure for o in  current_forward_program)

                                        if not REQUIRE_UPDATE_PROGRAM or already_mutated_value:
                                            add_current_program_to_results_list()

                                elif program_construction_stage == UPDATE_STAGE:
                                    current_update_program.append(
                                        operation_instance)

                                    # Note: We do NOT consider REQUIRE_UPDATE_PROGRAM=False here, because we still
                                    # only want to add new update programs if they actually do something
                                    is_full_program = operation.mutates_datastructure and program_uses_all_intermediates(
                                        program_construction_stage)
                                    if is_full_program:
                                        add_current_program_to_results_list()

                                # Recurse
                                recursive_search(
                                    new_min_program_number,
                                    new_program_construction_stage,
                                    recursion_depth + 1)

                                # Backtrack
                                if program_construction_stage == FORWARD_STAGE:
                                    current_forward_program.remove(
                                        operation_instance)
                                elif program_construction_stage == UPDATE_STAGE:
                                    current_update_program.remove(
                                        operation_instance)

                                remove_graph_node(operation_instance, output_type)
                                existing_program_numbers.remove(
                                    operation_program_number)

                                for i in input_tuple:
                                    i.num_uses_in_program -= 1

        for var in inputs + data_structures + optimizers:
            variables.append(var)
            add_graph_node(var, var.var_type)

        for var in inputs:
            assert not var.is_data_structure and not var.is_optimizer

        for var in data_structures:
            assert var.is_data_structure and not var.is_optimizer
            t = type(var.var_type)
            if t not in data_structures_by_type:
                data_structures_by_type[t] = []
            data_structures_by_type[t].append(var)

        for var in optimizers:
            assert var.is_optimizer and not var.is_data_structure

        for p in initial_program_list:
            add_program_to_results_list(p)

        recursive_search("")

        print(
            f"Found {len(programs)} valid programs " +
            f"in {time.time() - start_time} seconds. " +
            f"Reached target output {num_searches_that_reached_desired_output} times. " + 
            f"Num duplicates found: {duplicates_found}.")

        return programs


def numbers_are_increasing(lst: List[int]):
    prev = None
    for i in lst:
        if prev is not None and i > prev:
            return False
        prev = i
    return True

def main():
    parser = mlca.helpers.config.argparser()
    args = parser.parse_args()

    operations_list_name = args.experiment_id
    operations_set_params = operations_list.OperationsSetList.get(operations_list_name)

    with mlca.helpers.config.DefaultDevice(mlca.helpers.config.get_device_and_set_default()):
        for MAX_TOTAL_PROGRAM_LENGTH in range(
            operations_set_params.MIN_PROGRAM_LENGTH,
            operations_set_params.MAX_PROGRAM_LENGTH + 1
        ):
            print()
            print("-----------------------")
            print("Search MAX_TOTAL_PROGRAM_LENGTH =", MAX_TOTAL_PROGRAM_LENGTH)
            print(f"Operation list = {args.experiment_id}")
            print("-----------------------")

            inputs, data_structures, optimizers, initial_program_list = operations_set_params.INITIAL_VARIABLES_FN()

            programs = search(
                inputs, data_structures, optimizers, initial_program_list,
                operations_set_params,
                operations_set_params.REQUIRE_UPDATE_PROGRAM,
                MAX_TOTAL_PROGRAM_LENGTH,
            )

            print(f"Found {len(programs)} programs of max length {MAX_TOTAL_PROGRAM_LENGTH}")

            with open(f'pickles/programs_{operations_list_name}_len_{MAX_TOTAL_PROGRAM_LENGTH}.pickle', 'wb') as f:
                pickle.dump((programs, inputs, data_structures, optimizers), f)

if __name__ == "__main__":
    main()