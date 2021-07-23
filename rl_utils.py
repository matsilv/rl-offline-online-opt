from gym import Env
from gym.spaces import Box
import akro
import numpy as np
import pandas as pd
import random
from gurobipy import Model, GRB
from OnlineHeuristic import solve
from tabulate import tabulate
from datetime import datetime, timedelta

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
                 realizations,
                 cGrid,
                 shift):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param realizations: pandas.Dataframe; real PV and Load.
        :param cGrid: numpy.array; cGrid values.
        :param shift: numpy.array; shift values.
        """

        # Number of timesteps in 1 hour
        self.n = 96

        # NOTE: this must be 1 because we solve once instance at a time
        self.mrT = 1

        # These are variables related to the optimization model
        self.predictions = predictions
        self.realizations = realizations
        self.predictions = instances_preprocessing(self.predictions)
        self.realizations = instances_preprocessing(self.realizations)
        self.cGrid = cGrid
        self.shift = shift
        self.capMax = 1000
        self.inCap = 800
        self.cDiesel = 0.054
        self.pDieselMax = 1200

        # NOTE: here we define the observation and action spaces
        self.observation_space = Box(low=0, high=np.inf, shape=(self.n * 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32)

        # NOTE: we randomly choose an instance
        assert len(self.predictions) == len(self.realizations), "Predictions and realizations must have the same length"
        assert self.predictions.index.equals(self.realizations.index)
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())

        self._create_instance_variables()

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        # predicted PV for the current instance
        self.pRenPVpred = [self.predictions['PV(kW)'][self.mr] for i in range(self.mrT)]
        self.pRenPVpred = np.asarray(self.pRenPVpred)

        # predicted Load for the current instance
        self.tot_cons_pred = [self.predictions['Load(kW)'][self.mr] for i in range(self.mrT)]
        self.tot_cons_pred = np.asarray(self.tot_cons_pred)

        # real PV for the current instance
        self.pRenPVreal = [self.realizations['PV(kW)'][self.mr] for i in range(self.mrT)]
        self.pRenPVreal = np.asarray(self.pRenPVreal)

        # real Load for the current instance
        self.tot_cons_real = [self.realizations['Load(kW)'][self.mr] for i in range(self.mrT)]
        self.tot_cons_real = np.asarray(self.tot_cons_real)

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

        observations = np.concatenate((self.pRenPVpred.copy(), self.tot_cons_pred.copy()), axis=1)
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

    def _define_obj_function(self, model, c_virt, index):
        """
        Given a model, define the VPP cost function for a given timestep.
        :param model: gurobipy.Model; variables and constraints of the VPP model.
        :param c_virt: numpy.array; if not None, the virtual costs are multiplied to the output eletricity
                                    from the storage.
        :param index: int; the timestep.
        :return:
        """

        # Objective function
        # NOTE: the action are the virtual costs associated with the storage
        if c_virt is None:
            storage_cost = self.cGrid
        else:
            storage_cost = c_virt

        obf = (self.cGrid[index] * model.getVarByName['pGridOut_' + str(index)] +
               self.cDiesel * model.getVarByName['pDiesel_' + str(index)] +
               storage_cost[index] * model.getVarByName['pStorageIn_' + str(index)] -
               self.cGrid[index] * model.getVarByName['pGridIn_' + str(index)])

        return obf

    def _solve(self, c_virt=None):
        """
        Solve the optimization model with the greedy heuristic. If the virtual costs are set then we compute the virtual
        objective function, otherwise the real objective function is computed.
        :param c_virt: numpy.array of shape (num_timesteps, ); the virtual costs.
        :return: gurobipy.Model the cost of the solution found by the heuristic and a dictionary with the found
                                 solution.
        """

        # NOTE: check variables initialization
        assert self.mr is not None, "Instance index must be initialized"
        assert self.cGrid is not None, "cGrid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.pRenPVreal is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        cap, pDiesel, pStorageIn, pStorageOut, pGridIn, pGridOut, tilde_cons = [[None] * self.n for _ in range(7)]

        # Per-timestep storage capacitance
        capX = self.inCap

        # Save all the optimization models in a list
        models = []

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

            # NOTE: the heuristic uses the real load
            tilde_cons[i] = (self.shift[i] + self.tot_cons_real[j][i])

            ####################
            # Model constraints
            ####################

            # power balance constraint
            # NOTE: the heuristic uses the real PV
            mod.addConstr((self.pRenPVreal[j][i] + pStorageOut[i] + pGridOut[i] + pDiesel[i] -
                           pStorageIn[i] - pGridIn[i] == tilde_cons[i]), "Power balance")

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
            obf = self._define_obj_function(mod, c_virt, i)
            mod.setObjective(obf)

            solve(mod)

            models.append(mod)

        return models

    def _compute_real_cost(self, models):
        """
        Given a solution, compute the real costs.
        :param solution: dict; a dictionary with the solution.
        :return: float; the real cost of the given solution.
        """

        cost = 0

        for timestep in enumerate(models):
            model = models[timestep]
            optimal_pGridOut = model.getVarByName['pGridOut_' + str(timestep)]
            optimal_pDiesel = model.getVarByName['pDiesel_' + str(timestep)]
            optimal_pStorageIn = model.getVarByName['pStorageIn_' + str(timestep)]
            optimal_pGridIn = model.getVarByName['pGridIn_' + str(timestep)]
            cost += (self.cGrid[timestep] * optimal_pGridOut + self.cDiesel * optimal_pDiesel +
                     self.cGrid[timestep] * optimal_pStorageIn - self.cGrid[timestep] * optimal_pGridIn)

        return cost

    def step(self, action):
        """
        This is a step performed in the environment: the virtual costs are set by the agent and then the total cost
        (the reward) is computed.
        :param action: numpy.array of shape (num_timesteps, ); virtual costs for each timestep.
        :return: dict, float, boolean, dict; the observations, the reward, a boolean that is True if the episode is
                                             ended, additional information.
        """

        # Solve the optimization model with the virtual costs
        models = self._solve(action)

        # NOTE: the reward is the negative real cost cost
        reward = -self._compute_real_cost(models)

        # NOTE: the episode has a single timestep
        done = True

        # self._render_solution(objFinal, runFinal)

        observations = self._get_observations()

        return observations, reward, done, {}

    def reset(self):
        """
        When we reset the environment we randomly choose another instance.
        :return: numpy.array; pv and load values for the current instance.
        """

        self._clear()

        # NOTE: we randomly choose an instance
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())
        self._create_instance_variables()
        return self._get_observations()

    def render(self, mode='ascii'):
        """
        Simple rendering of the environment.
        :return:
        """

        timestamps = timestamps_headers(self.n)
        print('\nAction')
        print(tabulate(self.action, headers=timestamps, tablefmt='pretty'))
        print('\nPredicted PV(kW)')
        print(tabulate(self.pRenPVpred, headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(self.tot_cons_pred, headers=timestamps, tablefmt='pretty'))
        print('Real PV(kW)')
        print(tabulate(self.pRenPVreal, headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(self.tot_cons_real, headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass


########################################################################################################################


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
                 realizations,
                 cGrid,
                 shift):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param realizations: pandas.Dataframe; real PV and Load.
        :param cGrid: numpy.array; cGrid values.
        :param shift: numpy.array; shift values.
        """

        # Number of timesteps in 1 hour
        self.n = 96

        # NOTE: this must be 1 because we solve once instance at a time
        self.mrT = 1

        self.predictions = predictions
        self.realizations = realizations
        self.predictions = instances_preprocessing(self.predictions)
        self.realizations = instances_preprocessing(self.realizations)
        self.cGrid = cGrid
        self.shift = shift

        # NOTE: here we define the observation and action spaces
        self.observation_space = Box(low=0, high=np.inf, shape=(self.n * 2 + 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # NOTE: we randomly choose an instance
        assert len(self.predictions) == len(self.realizations), \
            "Predictions and realizations must have the same length"
        assert self.predictions.index.equals(self.realizations.index), \
            "Predictions and realizations musy have the same index"
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())

        self._create_instance_variables()

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance. Initialize timestep, storage usage, cost and
        execution time and solutions auxiliary variables.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        # predicted PV for the current instance
        self.pRenPVpred = [self.predictions['PV(kW)'][self.mr] for i in range(self.mrT)]
        self.pRenPVpred = np.asarray(self.pRenPVpred)

        # predicted Load for the current instance
        self.tot_cons_pred = [self.predictions['Load(kW)'][self.mr] for i in range(self.mrT)]
        self.tot_cons_pred = np.asarray(self.tot_cons_pred)

        # real PV for the current instance
        self.pRenPVreal = [self.realizations['PV(kW)'][self.mr] for i in range(self.mrT)]
        self.pRenPVreal = np.asarray(self.pRenPVreal)

        # real Load for the current instance
        self.tot_cons_real = [self.realizations['Load(kW)'][self.mr] for i in range(self.mrT)]
        self.tot_cons_real = np.asarray(self.tot_cons_real)

        # Initialize the timestep, storage usage and solution cost
        self.timestep = 0
        self.storage_usage = 800
        self.cost = 0
        self.listc = np.empty((self.mrT, self.n))
        self.objX = np.empty((self.mrT, self.n))

        # These are auxiliary variables that are used to store the runtime and the solution cost
        self.runList = []
        self.runtime = 0
        self.objList = []
        self.objFinal = 0

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
        Return predicted pv and load values, storage usage and initial cost, as a single array.
        :return: numpy.array; observations for the current instance.
        """

        observations = np.concatenate((self.pRenPVpred.copy(), self.tot_cons_pred.copy()), axis=1)
        observations = np.squeeze(observations)
        observations = np.append(observations, [self.storage_usage, self.cost])

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
        self.storage_usage = 800
        self.cost = 0
        self.runList = []
        self.runtime = 0
        self.objList = []
        self.objFinal = 0
        self.listc = np.empty((self.mrT, self.n))
        self.objX = np.empty((self.mrT, self.n))

    def step(self, action):
            """
            This is a step performed in the environment: a single step virtual cost is set by the agent.
            :param action: numpy.array of shape (1, ); virtual cost for the current timestep.
            :return: numpy.array, float, boolean, dict; the observations, the reward, a boolean that is True if the episode
                                                        is ended, additional information.
            """

            # NOTE: check variables initialization
            assert self.mr is not None, "Instance index must be initialized before the step function"
            assert self.cGrid is not None, "cGrid must be initialized before the step function"
            assert self.shift is not None, "shifts must be initialized before the step function"
            assert self.pRenPVreal is not None, "Real PV values must be initialized before the step function"
            assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"
            assert self.pRenPVpred is not None, "Predicted PV values must be initialize before the step function"
            assert self.tot_cons_pred is not None, "Predicted Load values must be initialized before the step function"

            # Enforce action space
            # NOTE: we MUST copy the action before modifying it
            c_virt = action.copy()
            c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)

            # NOTE: this is a copy and paste of the heur() method except for the i variable in the inner loop that is
            #  replaced by the current timestep

            # price data from GME
            cGridS = np.mean(self.cGrid)

            # capacities, bounds, parameters and prices
            # mrT = 1
            objTot = [None] * self.mrT
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, cap, change, phi, notphi, pDiesel, pStorageIn, pStorageOut, pGridIn, pGridOut, tilde_cons = [
                None for i in range(20)]
            capMax = 1000

            # NOTE: the initial storage usage is the one of the last timestep
            inCap = self.storage_usage
            capX = inCap
            cDiesel = 0.054
            cRU = 0.35
            pDieselMax = 1200
            runtime = 0
            phiX = 0
            solutions = np.zeros((self.mrT, self.n, 9))

            # if you want to run more than one instance at a time mrT != 1
            for j in range(self.mrT):
                # FIXME: here we simply replace the i variable with the current timestep but a refactoring is required
                #  since the lists can be replaced by scalar values

                i = self.timestep
                # create a model
                mod = Model()

                # build variables and define bounds
                pDiesel = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_" + str(i))
                pStorageIn = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_" + str(i))
                pStorageOut = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_" + str(i))
                pGridIn = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridIn_" + str(i))
                pGridOut = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridOut_" + str(i))
                cap = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))
                # change[i] = mod.addVar(vtype=GRB.INTEGER, name="change")
                # phi[i] = mod.addVar(vtype=GRB.BINARY, name="phi")
                # notphi[i] = mod.addVar(vtype=GRB.BINARY, name="notphi")

                #################################################
                # Shift from Demand Side Energy Management System
                #################################################

                # NOTE: the heuristic uses the real load
                tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[j][self.timestep])

                ####################
                # Model constraints
                ####################

                # more sophisticated storage constraints
                # mod.addConstr(notphi[i]==1-phi[i])
                # mod.addGenConstrIndicator(phi[i], True, pStorageOut[i], GRB.LESS_EQUAL, 0)
                # mod.addGenConstrIndicator(notphi[i], True, pStorageIn[i], GRB.LESS_EQUAL, 0)

                # power balance constraint
                # NOTE: the heuristic uses the real PV
                mod.addConstr(
                    (self.pRenPVreal[j][self.timestep] + pStorageOut + pGridOut + pDiesel - pStorageIn - pGridIn ==
                     tilde_cons), "Power balance")

                # Storage cap
                mod.addConstr(cap == capX + pStorageIn - pStorageOut)
                mod.addConstr(cap <= capMax)

                mod.addConstr(pStorageIn <= capMax - (capX))
                mod.addConstr(pStorageOut <= capX)

                mod.addConstr(pStorageIn <= 200)
                mod.addConstr(pStorageOut <= 200)

                # Diesel and Net cap
                mod.addConstr(pDiesel <= pDieselMax)
                mod.addConstr(pGridIn <= 600)

                # Storage mode change
                # mod.addConstr(change[i]>=0)
                # mod.addConstr(change[i]>= (phi[i] - phiX))
                # mod.addConstr(change[i]>= (phiX - phi[i]))

                # Objective function
                # NOTE: the action are the virtual cost associated with the storage
                obf = (self.cGrid[self.timestep] * pGridOut + cDiesel * pDiesel +
                       self.cGrid[self.timestep] * pStorageIn - self.cGrid[self.timestep] * pGridIn)
                # for using storage constraints for mode change we have to add cRU*change in the objective function

                mod.setObjective(obf)

                solve(mod)

                # extract x values
                a2 = pDiesel.X
                a4 = pStorageIn.X
                a5 = pStorageOut.X
                # NOTE: the heuristic uses the real PV
                a3 = self.pRenPVreal[j][self.timestep]
                a6 = pGridIn.X
                a7 = pGridOut.X
                a8 = cap.X
                a1 = tilde_cons
                self.objX[j][self.timestep] = mod.objVal
                a9 = self.cGrid[self.timestep]
                capX = cap.x
                self.listc[j][self.timestep] = capX
                # phiX = phi[i].x

                solutions[j][self.timestep] = [mod.objVal, a3, a1, capX, a2, a4, a5, a6, a7]

                a10 = self.shift
                data = np.array([a1, a2, a3, a9, a6, a7, a4, a5, a8, a10])

                # Keep track of the cost at each timestep
                self.objList.append((self.objX[j][self.timestep]))
                self.runList.append(mod.Runtime * 60)
                self.runtime += mod.Runtime * 60

                # NOTE: update the storage usage
                self.storage_usage = capX

            # NOTE: the reward is the negative cost
            reward = -self.objX[j][self.timestep]

            # NOTE: increase the timestep and, if it is the last one, terminate the episode
            self.timestep += 1
            if self.timestep == self.n:
                done = True

            else:
                done = False

            observations = self._get_observations()

            return observations, reward, done, {}

    def reset(self):
        """
        When we reset the environment we randomly choose another instance.
        :return: numpy.array; pv and load values for the current instance.
        """

        self._clear()

        # NOTE: we randomly choose an instance
        self.mr = random.randint(self.predictions.index.min(), self.predictions.index.max())
        self._create_instance_variables()
        return self._get_observations()

    def render(self, mode='ascii'):
        """
        Simple rendering of the environment.
        :return:
        """

        timestamps = timestamps_headers(self.n)
        print('Predicted PV(kW)')
        print(tabulate(self.pRenPVpred, headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(self.tot_cons_pred, headers=timestamps, tablefmt='pretty'))
        print('Real PV(kW)')
        print(tabulate(self.pRenPVreal, headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(self.tot_cons_real, headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass



########################################################################################################################


