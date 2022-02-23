# Author: Mattia Silvestri

"""
    Utility methods for the algorithm configuration use case.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.utility import instances_preprocessing

########################################################################################################################


def load_contigency_dataset(loadpath):
    df = pd.read_csv(loadpath)
    df = instances_preprocessing(df)

    experiments = dict()
    instances = set()

    for idx, row in df.iterrows():
        exp = Experiment(n_traces=row['nTraces'],
                         pv=row['PV(kW)'],
                         load=row['Load(kW)'],
                         sol=row['sol(keuro)'],
                         time=row['time(sec)'],
                         memory=row['memAvg(MB)'])

        if exp.key not in experiments.keys():
            experiments[exp.key] = exp

        instances.add(exp.instance_key)

    return experiments, instances

########################################################################################################################


class Experiment:
    """
    Class representing a single experiment.
    Attributes:
        key: string; an Experiment is uniquely identified by its pv, load and number of traces.
        instance_key: string; an instance is uniquely identified by its pv and load.
        n_traces: int; the number of employed traces.
        pv: numpy.array; the photovoltaic production.
        load: numpy.array; the user load demand.
        sol: float; the solution cost.
        time: float; time required to find the optimal solution.
        :avg_mem: float; the average memory required to solve the instance.
    """
    def __init__(self, n_traces, pv, load, sol, time, memory):
        assert isinstance(pv, (np.ndarray, list)), "pv must be a numpy.array of a list"
        assert isinstance(load, (np.ndarray, list)), "load must be a numpy.array of a list"

        self._n_traces = n_traces

        if isinstance(pv, list):
            self._pv = np.asarray(pv)
        else:
            self._pv = pv

        if isinstance(load, list):
            self._load = np.asarray(load)
        else:
            self._load = load

        self._sol = sol
        self._time = time
        self._avg_mem = memory

        pv_and_load = np.concatenate((pv, load), axis=0)
        self._instance_key = pv_and_load.tobytes()
        pv_load_and_n_traces = np.append(pv_and_load, self._n_traces)
        self._key = pv_load_and_n_traces.tobytes()

    @property
    def key(self):
        return self._key

    @property
    def instance_key(self):
        return self._instance_key

    @property
    def n_traces(self):
        return self._n_traces

    @property
    def pv(self):
        return self._pv

    @property
    def load(self):
        return self._load

    @property
    def sol(self):
        return self._sol

    @property
    def time(self):
        return self._time

    def __str__(self):
        res = ''
        res += f'N. traces: {self._n_traces}\n'
        res += f'PV (kW): {str(self._pv)}\n'
        res += f'Load(kW): {str(self._load)}\n'
        res += f'Solution cost (k€): {self._sol}\n'
        res += f'Time (sec): {self._time}\n'
        res += f'Average memory (GB): {self._avg_mem}\n'

        return res


########################################################################################################################


if __name__ == '__main__':
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