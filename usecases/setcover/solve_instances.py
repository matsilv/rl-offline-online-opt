# Author: Mattia Silvestri

from gurobipy import Model, GRB

########################################################################################################################


class MinSetCoverProblem:
    """
    Class for the Minimum Set Cover problem.
    """

    def __init__(self, instance):
        # Create the model
        self._problem = Model()

        # Add an integer variable for each product
        self._decision_vars = self._problem.addMVar(shape=instance.num_sets,
                                                    vtype=GRB.INTEGER,
                                                    lb=0,
                                                    name='Decision variables')

        # Demand satisfaction constraints
        self._problem.addConstr(instance.availability @ self._decision_vars >= instance.demands)

        # Set the objective function
        self._problem.setObjective(instance.costs @ self._decision_vars, GRB.MINIMIZE)

    def solve(self):
        self._problem.optimize()
        status = self._problem.status

        assert status == GRB.Status.OPTIMAL, "Solution is not optimal"

        solution = self._decision_vars.X
        obj_val = self._problem.objVal

        print_str = ""
        for idx in range(len(solution)):
            print_str += f'Set n.{idx}: {solution[idx]} - '
        print_str += f'\nSolution cost: {obj_val}'

        print(print_str)
