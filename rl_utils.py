from gym import Env
from gym.spaces import Box
import numpy as np
import random
from gurobipy import Model, GRB
from OnlineHeuristic import solve
from tabulate import tabulate
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os, shutil

########################################################################################################################

MIN_REWARD = -100000

########################################################################################################################


def timestamps_headers(num_timeunits):
    """
    Given a number of timeunits (in minutes), it provides a string representation of each timeunit.
    For example, if num_timeunits=96, the result is [00:00, 00:15, 00:30, ...].
    :param num_timeunits: int; the number of timeunits in a day.
    :return: list of string; list of timeunits.
    """

    start_time = datetime.strptime('00:00', '%H:%M')
    timeunit = 24 * 60 / num_timeunits
    timestamps = [start_time + idx * timedelta(minutes=timeunit) for idx in range(num_timeunits)]
    timestamps = ['{:02d}:{:02d}'.format(timestamp.hour, timestamp.minute) for timestamp in timestamps]

    return timestamps

########################################################################################################################


def instances_preprocessing(instances):
    """
    Convert PV and Load values from string to float.
    :param instances: pandas.Dataframe; PV and Load for each timestep and for every instance.
    :return: pandas.Dataframe; the same as the input dataframe but with float values instead of string.
    """

    assert 'PV(kW)' in instances.keys(), "PV(kW) must be in the dataframe columns"
    assert 'Load(kW)' in instances.keys(), "Load(kW) must be in the dataframe columns"

    # Instances pv from file
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    # Instances load from file
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))

    return instances

########################################################################################################################


def compare_cost(filepath1,
                 filepath2,
                 name1,
                 name2,
                 baseline=None):

    sns.set_style('darkgrid')

    rew1 = pd.read_csv(filepath1)['Extras/EpisodeRewardMean']
    rew2 = pd.read_csv(filepath2)['Extras/EpisodeRewardMean']
    rew1 = -rew1
    rew1[rew1 > 500] = np.nan

    rew2 = -rew2
    rew2[rew2 > 500] = np.nan

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d (k€)'))
    plt.title('Average episode reward', fontweight='bold')
    plt.xlabel('Epoch')

    rew1.name = name1
    rew2.name = name2

    rew1.plot()
    rew2.plot()

    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline mean cost')

    plt.legend()
    plt.show()


########################################################################################################################

class VPPEnv(Env):
    """
    Gym environment for the VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self,
                 predictions,
                 cGrid,
                 shift,
                 test_split,
                 test_mode,
                 instance_idx=None,
                 noise_std_dev=0.02,
                 savepath=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param cGrid: numpy.array; cGrid values.
        :param shift: numpy.array; shift values.
        :param test_split: float; fraction of the predictions to be used as test.
        :param test_mode: bool; True to use the test predictions, False to use the training ones.
        :param instance_idx: int; the index of the instance used for testing; it must be specified only when test_mode
                                  is true.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """

        # Assert that the index of the instance to be tested is selected when in testing mode
        if test_mode:
            assert instance_idx is not None, "If testing mode is selected then you must specify the instance index"

        # Set numpy random seed to ensure reproducibility
        np.random.seed(0)

        # Number of timesteps in one day
        self.n = 96

        # Standard deviation of the additive gaussian noise
        self.noise_std_dev = noise_std_dev

        # These are variables related to the optimization model
        self.predictions = predictions
        self.predictions = instances_preprocessing(self.predictions)
        self.cGrid = cGrid
        self.shift = shift
        self.capMax = 1000
        self.inCap = 800
        self.cDiesel = 0.054
        self.pDieselMax = 1200

        # Here we define the observation and action spaces
        self.observation_space = Box(low=0, high=np.inf, shape=(self.n * 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32)

        self.savepath = savepath

        # Split between training and test set
        split_index = int(len(self.predictions) * (1 - test_split))
        train_predictions = self.predictions.iloc[:split_index].copy()
        test_predictions = self.predictions.iloc[split_index:].copy()
        assert instance_idx in test_predictions.index.values, "Instance index not valid"
        test_predictions = test_predictions.loc[[instance_idx]]

        # predicted PV for the current instance
        pRenPVpred_train = np.array([np.array(x) for x in train_predictions['PV(kW)']])
        pRenPVpred_test = np.array([np.array(x) for x in test_predictions['PV(kW)']])
        self.max_pRenPVpred = np.max(pRenPVpred_train)

        # predicted Load for the current instance
        tot_cons_pred_train = np.array([np.array(x) for x in train_predictions['Load(kW)']])
        tot_cons_pred_test = np.array([np.array(x) for x in test_predictions['Load(kW)']])
        self.max_tot_cons_pred = np.max(tot_cons_pred_train)

        # Initialize the selected instances and their index range
        if test_mode:
            self.instances_indexes = np.arange(0, len(test_predictions))
            self.selected_pRenPVpred = pRenPVpred_test
            self.selected_tot_cons_pred = tot_cons_pred_test
        else:
            self.instances_indexes = np.arange(0, len(train_predictions), dtype=np.int32)
            self.selected_pRenPVpred = pRenPVpred_train
            self.selected_tot_cons_pred = tot_cons_pred_train

        self._create_instance_variables()

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.selected_tot_cons_pred is not None, "selected_tot_cons_pred must be initialized"
        assert self.selected_pRenPVpred is not None, "selected_pRenPVpred must be initialized"
        assert self.instances_indexes is not None, "instances_indexes must be initialized"

        # Randomly choose an instance and the corresponding predictions
        self.mr = np.random.choice(self.instances_indexes, size=1).item()
        self.pRenPVpred = self.selected_pRenPVpred[self.mr]
        self.tot_cons_pred = self.selected_tot_cons_pred[self.mr]

        # Loop until you find a realization which is feasible
        feasible = False

        while not feasible:
            # The real PV for the current instance is computed adding noise to the predictions
            noise = np.random.normal(0, self.noise_std_dev, self.n)
            self.pRenPVreal = self.pRenPVpred + self.pRenPVpred * noise

            # The real Load for the current instance is computed adding noise to the predictions
            noise = np.random.normal(0, self.noise_std_dev, self.n)
            self.tot_cons_real = self.tot_cons_pred + self.tot_cons_pred * noise

            _, feasible = self._solve(c_virt=np.zeros_like(self.cGrid))

            if not feasible:
                print('Found unfeasible realizations')

    def _render_solution(self, objFinal, runFinal):
        """
        Solution visualization.
        :param objFinal: list of float; the final cost for each instance.
        :param runFinal: list of float; the time required to solve each instance.
        :return:
        """

        print("\n============================== Solutions of Instance %d  =================================\n\n"
              % self.mr)

        print("The solution cost (in keuro) is: %s\n" % (str(np.mean(objFinal))))
        print("The runtime (in sec) is: %s\n" % (str(np.mean(runFinal))))

    def _get_observations(self):
        """
        Return predicted pv and load values as a single array.
        :return: numpy.array; pv and load values for the current instance.
        """

        observations = np.concatenate((self.pRenPVpred / self.max_pRenPVpred,
                                       self.tot_cons_pred / self.max_tot_cons_pred), axis=0)
        observations = np.squeeze(observations)

        return observations

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.pRenPVpred = None
        self.pRenPVreal = None
        self.tot_cons_pred = None
        self.tot_cons_real = None
        self.mr = None

    def _solve(self, c_virt):
        """
        Solve the optimization model with the greedy heuristic.
        :param c_virt: numpy.array of shape (num_timesteps, ); the virtual costs multiplied to output storage variable.
        :return: list of gurobipy.Model; a list with the solved optimization model.
        """

        # Check variables initialization
        assert self.mr is not None, "Instance index must be initialized"
        assert self.cGrid is not None, "cGrid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.pRenPVreal is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        cap, pDiesel, pStorageIn, pStorageOut, pGridIn, pGridOut, tilde_cons = [[None] * self.n for _ in range(7)]

        # Initialize the storage capacitance
        capX = self.inCap

        # Save all the optimization models in a list
        models = []

        # Loop for each timestep
        for i in range(self.n):
            # create a model
            mod = Model()

            # build variables and define bounds
            pDiesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_" + str(i))
            pStorageIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_" + str(i))
            pStorageOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_" + str(i))
            pGridIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridIn_" + str(i))
            pGridOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridOut_" + str(i))
            cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))

            #################################################
            # Shift from Demand Side Energy Management System
            #################################################

            # NOTE: the heuristic uses the real load and photovoltaic production

            tilde_cons[i] = (self.shift[i] + self.tot_cons_real[i])

            ####################
            # Model constraints
            ####################

            # power balance constraint
            mod.addConstr((self.pRenPVreal[i] + pStorageOut[i] + pGridOut[i] + pDiesel[i] -
                           pStorageIn[i] - pGridIn[i] == tilde_cons[i]), "Power_balance")

            # Storage cap
            mod.addConstr(cap[i] == capX + pStorageIn[i] - pStorageOut[i])
            mod.addConstr(cap[i] <= self.capMax)

            mod.addConstr(pStorageIn[i] <= self.capMax - capX)
            mod.addConstr(pStorageOut[i] <= capX)

            mod.addConstr(pStorageIn[i] <= 200)
            mod.addConstr(pStorageOut[i] <= 200)

            # Diesel and Net cap
            mod.addConstr(pDiesel[i] <= self.pDieselMax)
            mod.addConstr(pGridIn[i] <= 600)

            # for using storage constraints for mode change we have to add cRU*change in the objective function

            obf = (self.cGrid[i] * pGridOut[i] + self.cDiesel * pDiesel[i] +
                   c_virt[i] * pStorageIn[i] - self.cGrid[i] * pGridIn[i])
            mod.setObjective(obf)

            feasible = solve(mod)

            # FIXME: remove
            # mod.write(f'temp/model_{i}.lp')

            # If one of the timestep is not feasible, get out of the loop
            if not feasible:
                break

            models.append(mod)

            # Update the storage capacitance
            capX = cap[i].X

        return models, feasible

    def _compute_real_cost(self, models):
        """
        Given a list of models, one for each timestep, the method returns the real cost value.
        :param models: list of gurobipy.Model; a list with an optimization model for each timestep.
        :return: float; the real cost of the given solution.
        """

        assert len(models) == self.n

        cost = 0
        all_cost = []

        # Compute the total cost considering all the timesteps
        for timestep, model in enumerate(models):
            optimal_pGridOut = model.getVarByName('pGridOut_' + str(timestep)).X
            optimal_pDiesel = model.getVarByName('pDiesel_' + str(timestep)).X
            optimal_pGridIn = model.getVarByName('pGridIn_' + str(timestep)).X

            cost += (self.cGrid[timestep] * optimal_pGridOut + self.cDiesel * optimal_pDiesel
                     - self.cGrid[timestep] * optimal_pGridIn)
            all_cost.append(cost)

        return cost

    def step(self, action):
        """
        This is a step performed in the environment: the virtual costs are set by the agent and then the total cost
        (the reward) is computed.
        :param action: numpy.array of shape (num_timesteps, ); virtual costs for each timestep.
        :return: numpy.array, float, boolean, dict; the observations, the reward, a boolean that is True if the episode
                                                    is ended, additional information.
        """

        # Solve the optimization model with the virtual costs
        models, feasible = self._solve(action)

        if not feasible:
            reward = MIN_REWARD
            print('Unfeasible action performed by the agent')
        else:
            # The reward is the negative real cost
            reward = -self._compute_real_cost(models)

        # The episode has a single timestep
        done = True

        # self._render_solution(objFinal, runFinal)

        observations = self._get_observations()

        return observations, reward, done, {}

    def reset(self):
        """
        When we reset the environment we randomly choose another instance and we clear all the instance variables.
        :return: numpy.array; pv and load values for the current instance.
        """

        self._clear()

        # We randomly choose an instance
        self._create_instance_variables()
        return self._get_observations()

    def render(self, mode='ascii'):
        """
        Simple rendering of the environment.
        :return:
        """

        timestamps = timestamps_headers(self.n)
        print('\nPredicted PV(kW)')
        print(tabulate(np.expand_dims(self.pRenPVpred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal PV(kW)')
        print(tabulate(np.expand_dims(self.pRenPVreal, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_real, axis=0), headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass


########################################################################################################################


# FIXME: the class must be updated to support train/test split
class MarkovianVPPEnv(Env):
    """
    Gym environment for the Markovian version of the VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self,
                 predictions,
                 cGrid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param cGrid: numpy.array; cGrid values.
        :param shift: numpy.array; shift values.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """

        # Set numpy random seed to ensure reproducibility
        np.random.seed(0)

        # Number of timesteps in one day
        self.n = 96

        # Standard deviation of the additive gaussian noise
        self.noise_std_dev = noise_std_dev

        # These are variables related to the optimization model
        self.predictions = predictions
        self.predictions = instances_preprocessing(self.predictions)
        self.cGrid = cGrid
        self.shift = shift
        self.capMax = 1000
        self.inCap = 800
        self.cDiesel = 0.054
        self.pDieselMax = 1200

        # Here we define the observation and action spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n * 3 + 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # We randomly choose an instance
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())

        self.savepath = savepath

        self._create_instance_variables()

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        # Set the timestep
        self.timestep = 0

        # Set the cumulative cost
        self.cumulative_cost = 0

        # Initialize the storage
        self.storage = self.inCap

        # predicted PV for the current instance
        self.pRenPVpred = self.predictions['PV(kW)'][self.mr]
        self.pRenPVpred = np.asarray(self.pRenPVpred)

        # predicted Load for the current instance
        self.tot_cons_pred = self.predictions['Load(kW)'][self.mr]
        self.tot_cons_pred = np.asarray(self.tot_cons_pred)

        # Loop until you find a realization which is feasible
        feasible = False

        while not feasible:
            # The real PV for the current instance is computed adding noise to the predictions
            noise = np.random.normal(0, self.noise_std_dev, self.n)
            self.pRenPVreal = self.pRenPVpred + self.pRenPVpred * noise

            # The real Load for the current instance is computed adding noise to the predictions
            noise = np.random.normal(0, self.noise_std_dev, self.n)
            self.tot_cons_real = self.tot_cons_pred + self.tot_cons_pred * noise

            done = False
            feasible = True

            while not done:
                observations, reward, done, _ = self.step(action=0, agent_step=False)
                if reward == MIN_REWARD:
                    feasible = False
                    break

            if not feasible:
                print('Found infeasible realizations\n')

        # Reset time-dependent variables
        self.storage = self.inCap
        self.timestep = 0

    def _render_solution(self, objFinal, runFinal):
        """
        Solution visualization.
        :param objFinal: list of float; the final cost for each instance.
        :param runFinal: list of float; the time required to solve each instance.
        :return:
        """

        print("\n============================== Solutions of Instance %d  =================================\n\n"
              % self.mr)

        print("The solution cost (in keuro) is: %s\n" % (str(np.mean(objFinal))))
        print("The runtime (in sec) is: %s\n" % (str(np.mean(runFinal))))

    def _get_observations(self):
        """
        Return predicted pv and load values as a single array.
        :return: numpy.array; pv and load values for the current instance.
        """

        observations = np.concatenate((self.pRenPVpred.copy(), self.tot_cons_pred.copy()), axis=0)
        one_hot_timestep = np.zeros(shape=(self.n, ))
        one_hot_timestep[int(self.timestep)] = 1
        observations = np.concatenate((observations, one_hot_timestep), axis=0)
        observations = np.append(observations, self.storage)
        observations = np.append(observations, self.cumulative_cost)
        observations = np.squeeze(observations)

        return observations

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.pRenPVpred = None
        self.pRenPVreal = None
        self.tot_cons_pred = None
        self.tot_cons_real = None
        self.mr = None
        self.storage = self.inCap
        self.timestep = 0
        self.cumulative_cost = 0

    def _solve(self, c_virt):
        """
        Solve the optimization model with the greedy heuristic.
        :param c_virt: numpy.array of shape (num_timesteps, ); the virtual costs multiplied to output storage variable.
        :return: list of gurobipy.Model; a list with the solved optimization model.
        """

        # Check variables initialization
        assert self.mr is not None, "Instance index must be initialized"
        assert self.cGrid is not None, "cGrid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.pRenPVreal is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"
        assert self.storage is not None, "Storage variable must be initialized"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        c_virt = np.squeeze(c_virt)

        # create a model
        mod = Model()

        # build variables and define bounds
        pDiesel = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel")
        pStorageIn = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn")
        pStorageOut = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut")
        pGridIn = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridIn")
        pGridOut = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridOut")
        cap = mod.addVar(vtype=GRB.CONTINUOUS, name="cap")

        #################################################
        # Shift from Demand Side Energy Management System
        #################################################

        # NOTE: the heuristic uses the real load and photovoltaic production

        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        ####################
        # Model constraints
        ####################

        # power balance constraint
        mod.addConstr((self.pRenPVreal[self.timestep] + pStorageOut + pGridOut + pDiesel -
                       pStorageIn - pGridIn == tilde_cons), "Power_balance")

        # Storage cap
        mod.addConstr(cap == self.storage + pStorageIn - pStorageOut)
        mod.addConstr(cap <= self.capMax)

        mod.addConstr(pStorageIn <= self.capMax - self.storage)
        mod.addConstr(pStorageOut <= self.storage)

        mod.addConstr(pStorageIn <= 200)
        mod.addConstr(pStorageOut <= 200)

        # Diesel and Net cap
        mod.addConstr(pDiesel <= self.pDieselMax)
        mod.addConstr(pGridIn <= 600)

        # for using storage constraints for mode change we have to add cRU*change in the objective function

        obf = (self.cGrid[self.timestep] * pGridOut + self.cDiesel * pDiesel +
               c_virt * pStorageIn - self.cGrid[self.timestep] * pGridIn)
        mod.setObjective(obf)

        feasible = solve(mod)

        # Update the storage capacitance
        if feasible:
            self.storage = cap.X

        return mod, feasible

    def _compute_real_cost(self, model):
        """
        Given a list of models, one for each timestep, the method returns the real cost value.
        :param model: gurobipy.Model; optimization model for the current timestep.
        :return: float; the real cost of the current timestep.
        """

        optimal_pGridOut = model.getVarByName('pGridOut').X
        optimal_pDiesel = model.getVarByName('pDiesel').X
        optimal_pGridIn = model.getVarByName('pGridIn').X
        optimal_pStorageIn = model.getVarByName('pStorageIn').X
        optimal_pStorageOut = model.getVarByName('pStorageOut').X
        optimal_cap = model.getVarByName('cap').X

        cost = (self.cGrid[self.timestep] * optimal_pGridOut + self.cDiesel * optimal_pDiesel
                - self.cGrid[self.timestep] * optimal_pGridIn)

        return cost

    def step(self, action, agent_step=True):
        """
        This is a step performed in the environment: the virtual costs are set by the agent and then the total cost
        (the reward) is computed.
        :param action: numpy.array of shape (num_timesteps, ); virtual costs for each timestep.
        :return: numpy.array, float, boolean, dict; the observations, the reward, a boolean that is True if the episode
                                                    is ended, additional information.
        """

        # Solve the optimization model with the virtual costs
        models, feasible = self._solve(action)

        if not feasible:
            reward = MIN_REWARD
            if agent_step:
                print('Performed infeasible action\n')
        else:
            # The reward is the negative real cost
            reward = -self._compute_real_cost(models)

        # Update the cumulative cost
        self.cumulative_cost -= reward

        # self._render_solution(objFinal, runFinal)

        observations = self._get_observations()

        # Update the timestep
        self.timestep += 1

        if self.timestep == self.n or not feasible:
            done = True
        elif self.timestep < self.n:
            done = False
        else:
            raise Exception(f"Timestep cannot be greater than {self.n}")

        return observations, reward, done, {'feasible': feasible}

    def reset(self):
        """
        When we reset the environment we randomly choose another instance and we clear all the instance variables.
        :return: numpy.array; pv and load values for the current instance.
        """

        self._clear()

        # We randomly choose an instance
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())
        self._create_instance_variables()
        return self._get_observations()

    def render(self, mode='ascii'):
        """
        Simple rendering of the environment.
        :return:
        """

        timestamps = timestamps_headers(self.n)
        print('\nPredicted PV(kW)')
        print(tabulate(np.expand_dims(self.pRenPVpred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal PV(kW)')
        print(tabulate(np.expand_dims(self.pRenPVreal, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_real, axis=0), headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass

########################################################################################################################


