"""
A representation of a synthesized program.
Programs consist of a "forward" phase and an "update" phase. Each phase
is represented by a list of operations.
"""

import mlca.operations as operations
import mlca.program_types as program_types
from graphviz import Digraph
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import namedtuple

# ProgramWithId exists only for backwards compatibility. Use Program. 
ProgramWithId = namedtuple(
    'ProgramWithId', ('forward_program', 'update_program', 'program_id'))
ProgramId = int

@dataclass
class Program:
    forward_program: List[operations.Operation]
    update_program: List[operations.Operation]

    input_variables: Dict[str, operations.Variable]
    data_structure_variables: Dict[str, operations.Variable]
    optimizer_variables: Dict[str, operations.Variable]

    program_id: ProgramId
    name: Optional[str] = None

    """
    Execute this program (forward + update phases) and return the output
    of the last variable in the forward phase.
    """
    def execute(
        self, 
        input_values: Dict[str, Any], 
        data_structure_values: Dict[str, Any], 
        optimizer_values: Dict[str, Any], 
        profiler=None, print_on_error=True, i_episode=None):
        
        from mlca.executor import _execute_program 
        return _execute_program(
            self.forward_program + self.update_program,
            input_values,
            data_structure_values,
            optimizer_values,
            True,
            print_on_error,
            profiler=profiler,
            i_episode=i_episode
        )[self.forward_program[-1]]

    def visualize_as_graph(self, debug_text:Optional[str]=None):
        filename = self.name if self.name else str(self.program_id)

        def omit_variable(operation: operations.Operation):
            # TODO: Do this in a more general way
            return not (type(operation) == operations.Variable and operation.name == "Adam")

        g = Digraph(f"tmp/program_{filename}", format='png')

        variables: List[operations.Variable] = []
        for op in self.forward_program + self.update_program:
            variables.extend(op.inputs)
            variables.append(op)
        variables = list(filter(omit_variable, variables))

        gets_gradients: Dict[operations.Operation, bool] = {}
        for op in reversed(self.forward_program + self.update_program):
            if op.generates_backward_gradients or op in gets_gradients:
                gets_gradients[op] = True
                for i, inp in enumerate(op.inputs):
                    if op.propagates_gradients_to_input_i(i) and not (type(inp) == operations.Variable and not inp.is_data_structure):
                        gets_gradients[inp] = True

        var_to_name = self._variable_names(list(set(variables)))

        def node_id(node: operations.Operation):
            return var_to_name[node]

        def node_name(node: operations.Operation):
            if isinstance(node, operations.Variable): 
                if node.name is None and node.short_name is None:
                    return type(node.cached_output_type).__name__
                elif hasattr(node, "short_name") and node.short_name is not None:
                    return f"< <B> {str(node.short_name)} </B> >"
                else:
                    return f"< <B> {str(node.name)} </B> >"
            else: 
                nodename = node.short_name if hasattr(
                    node, 'short_name') else type(node).__name__
                output_type = type(node.cached_output_type).__name__ if not hasattr(
                    node.cached_output_type, 'short_name') else node.cached_output_type.short_name
                return f"< <B>{str(nodename)}</B> >"
                # <BR/>" \
                #     + f"{output_type} >"  # ({var_to_name[node]})

        if debug_text:
            g.node("debug_text", label=str(debug_text))

        for node in set(variables):
            with g.subgraph() as c:
                peripheries = 2 if program_types.equal_or_supertype(
                    type(node.cached_output_type), program_types.List) else 1
                shape = 'box'

                if isinstance(node, operations.Variable):
                    is_data_structure = node.name is None or node.is_data_structure or program_types.equal_or_supertype(type(node.output_type), program_types.Optimizer)
                    color = 'lightgray' if is_data_structure else 'lightblue'
                    bordercolor = 'violet' if node in gets_gradients else ""
                    c.attr('node', shape=shape, style='filled', fillcolor=color, color=bordercolor)
                elif node == self.forward_program[-1]:
                    c.attr('node', shape=shape, style='filled', color='green')
                elif node.generates_backward_gradients:
                    c.attr('node', shape=shape, style='filled', color='violet')
                elif node.mutates_datastructure:
                    c.attr('node', shape=shape, style='filled', color='plum1')
                else:
                    t = type(node.cached_output_type)
                    type_colors = {
                        program_types.FeatureVector: "orange",
                        program_types.RealNumber: "green",
                    }

                    bordercolor = 'black'
                    for supertype in type_colors:
                        if program_types.equal_or_supertype(t, supertype) or \
                                (program_types.equal_or_supertype(t, program_types.List)
                                    and program_types.equal_or_supertype(t.list_contents_type, supertype)):
                            bordercolor = type_colors[supertype]

                    c.attr('node', shape=shape, peripheries=str(
                        peripheries), color=bordercolor) 

                c.node(node_id(node), label=node_name(node))

        for node in filter(omit_variable, self.forward_program + self.update_program):
            for input_num, input_node in enumerate(filter(omit_variable, node.inputs)):
                with g.subgraph() as c:
                    if node in gets_gradients and node.propagates_gradients_to_input_i(input_num):
                        c.attr('edge', color='violet')

                    if hasattr(node, "input_program_names"):
                        label = node.input_program_names[input_num]
                    else:
                        label = None

                    c.edge(
                        node_id(input_node),
                        node_id(node),
                        label)

        g.render(view=True, cleanup=True)

    def __str__(self):
        variables: List[operations.Operation] = []
        for op in self.forward_program + self.update_program:
            variables.extend(op.inputs)

        var_to_name = self._variable_names(variables)
        
        output = []
        for operation in self.forward_program + [' # update only: '] + self.update_program:
            if type(operation) == str:
                output.append(operation)
            else:
                c = ", ".join(var_to_name[i] for i in operation.inputs)
                output.append(f"{var_to_name[operation]} = {operation.__class__.__name__}({c})".ljust(60)
                            + f" # {operation.cached_output_type.__class__.__name__}".ljust(30)
                            + f" {operation.input_set}")

        f = ", ".join([var_to_name[i] for i in self.forward_program])
        b = ", ".join([var_to_name[i] for i in self.update_program])
        output.append(f"program = Program([{f}], [{b}])")

        return "\n".join(output)

    def _variable_names(self, variables):
        var_to_name = {}

        name_num = 0
        for variable in variables + self.forward_program + self.update_program:
            if type(variable) is operations.Variable:
                var_name = variable.name
            else:
                var_name = chr(name_num + 97)
                name_num += 1

            var_to_name[variable] = var_name

        return var_to_name

    def initialize_program_structures(self, environment_class, policy):
        # NOTE: We always initialize all datastructures to avoid random intialization.

        if self.data_structure_variables is None:
            return None, None
        else: 
            data_structure_values = {
                d: d.output_type.create_empty(environment_class, policy) for d in self.data_structure_variables
            }

            optimizer_values = {
                o: o.output_type.create_empty(environment_class, data_structure_values) for o in self.optimizer_variables
            }

            return data_structure_values, optimizer_values

if __name__ == "__main__":
    import mlca.scripts.manually_evaluate_program as m

    disagreement, program_inputs, data_structures, optimizers = m.build_disagreement_program()
    inverse, program_inputs, data_structures, optimizers = m.build_inverse_program()
    rnd, program_inputs, data_structures, optimizers = m.build_random_network_distillation_program()
    
    best, program_inputs, data_structures, optimizers = m.build_best_program()
    fast, program_inputs, data_structures, optimizers = m.build_FAST_program()
    double_predict, program_inputs, data_structures, optimizers = m.build_double_predict_program()

    disagreement.visualize_as_graph()
    inverse.visualize_as_graph()
    rnd.visualize_as_graph()
    best.visualize_as_graph()
    fast.visualize_as_graph()
    double_predict.visualize_as_graph()
