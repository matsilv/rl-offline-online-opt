import matplotlib.pyplot as plt
import numpy as np
from .utility import load_contigency_dataset

########################################################################################################################


def main():
    experiments, instances = load_contigency_dataset(loadpath='data/CONT_trainDataset.csv')

    normalized_sol_dict = dict()
    for i in range(1, 101):
        normalized_sol_dict[i] = list()

    for inst in instances:
        n_traces = list()
        solutions = list()

        for exp in experiments.values():
            if inst == exp.instance_key:
                n_traces.append(exp.n_traces)
                solutions.append(exp.sol)

        best_sol = min(solutions)
        solutions = np.asarray(solutions) / best_sol

        for n_trac, sol in zip(n_traces, solutions):
            normalized_sol_dict[n_trac].append(sol)

    normalized_sol = np.array(list(normalized_sol_dict.values()))
    mean_sol = np.mean(normalized_sol, axis=1)
    std_sol = np.std(normalized_sol, axis=1)

    n_traces = list(normalized_sol_dict.keys())
    plt.plot(n_traces, mean_sol, color='b')
    plt.fill_between(n_traces, mean_sol - std_sol, mean_sol + std_sol, color='lightblue')
    plt.xlabel('# of traces')
    plt.title('Normalized solution cost')
    plt.show()
