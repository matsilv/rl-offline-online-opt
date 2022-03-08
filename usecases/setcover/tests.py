# Author: Mattia Silvestri

from usecases.setcover.generate_instances import MinSetCover
from usecases.setcover.solve_instances import MinSetCoverProblem

########################################################################################################################


def main():
    instance = MinSetCover(num_products=5, num_sets=3)
    print(instance)
    problem = MinSetCoverProblem(instance=instance)
    problem.solve()