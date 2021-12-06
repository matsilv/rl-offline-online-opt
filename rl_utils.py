from gym import Env
from gym.spaces import Box
import numpy as np
import random
from gurobipy import Model, GRB
from OnlineHeuristic import solve
from tabulate import tabulate
from utility import instances_preprocessing, timestamps_headers

########################################################################################################################

MIN_REWARD = -100000

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
                 c_grid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
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
        self.c_grid = c_grid
        self.shift = shift
        self.cap_max = 1000
        self.in_cap = 800
        self.c_diesel = 0.054
        self.p_diesel_max = 1200

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

        # predicted PV for the current instance
        self.p_ren_pv_pred = self.predictions['PV(kW)'][self.mr]
        self.p_ren_pv_pred = np.asarray(self.p_ren_pv_pred)

        # predicted Load for the current instance
        self.tot_cons_pred = self.predictions['Load(kW)'][self.mr]
        self.tot_cons_pred = np.asarray(self.tot_cons_pred)

        # The real PV for the current instance is computed adding noise to the predictions
        noise = np.random.normal(0, self.noise_std_dev, self.n)
        self.p_ren_pv_real = self.p_ren_pv_pred + self.p_ren_pv_pred * noise

        # The real Load for the current instance is computed adding noise to the predictions
        noise = np.random.normal(0, self.noise_std_dev, self.n)
        self.tot_cons_real = self.tot_cons_pred + self.tot_cons_pred * noise

    def step(self, action):
        """
        Step function of the Gym environment.
        :param action: numpy.array; agent's action.
        :return:
        """
        raise NotImplementedError()

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
        print(tabulate(np.expand_dims(self.p_ren_pv_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nPredicted Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_pred, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal PV(kW)')
        print(tabulate(np.expand_dims(self.p_ren_pv_real, axis=0), headers=timestamps, tablefmt='pretty'))
        print('\nReal Load(kW)')
        print(tabulate(np.expand_dims(self.tot_cons_real, axis=0), headers=timestamps, tablefmt='pretty'))

    def close(self):
        """
        Close the environment.
        :return:
        """
        pass


########################################################################################################################


class SingleStepVPPEnv(VPPEnv):
    """
    Gym environment for the single step version VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self,
                 predictions,
                 c_grid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """

        super(SingleStepVPPEnv, self).__init__(predictions, c_grid, shift, noise_std_dev, savepath)

        # Here we define the observation and action spaces
        self.observation_space = Box(low=0, high=np.inf, shape=(self.n * 2,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32)

    def _get_observations(self):
        """
        Return predicted pv and load values as a single array.
        :return: numpy.array; pv and load values for the current instance.
        """

        observations = np.concatenate((self.p_ren_pv_pred / np.max(self.p_ren_pv_pred),
                                       self.tot_cons_pred / np.max(self.tot_cons_pred)),
                                      axis=0)
        observations = np.squeeze(observations)

        return observations

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.p_ren_pv_pred = None
        self.p_ren_pv_real = None
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
        assert self.c_grid is not None, "c_grid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.p_ren_pv_real is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        cap, p_diesel, p_storage_in, p_storage_out, p_grid_in, p_grid_out, tilde_cons = \
            [[None] * self.n for _ in range(7)]

        # Initialize the storage capacitance
        cap_x = self.in_cap

        # Save all the optimization models in a list
        models = []

        # Loop for each timestep
        for i in range(self.n):
            # Create a Gurobi model
            mod = Model()

            # Build variables and define bounds
            p_diesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel_" + str(i))
            p_storage_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in_" + str(i))
            p_storage_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out_" + str(i))
            p_grid_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in_" + str(i))
            p_grid_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out_" + str(i))
            cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))

            # Shifts from Demand Side Energy Management System
            tilde_cons[i] = (self.shift[i] + self.tot_cons_real[i])

            # Power balance constraint
            mod.addConstr((self.p_ren_pv_real[i] + p_storage_out[i] + p_grid_out[i] + p_diesel[i] -
                           p_storage_in[i] - p_grid_in[i] == tilde_cons[i]), "Power_balance")

            # Storage capacity constraint
            mod.addConstr(cap[i] == cap_x + p_storage_in[i] - p_storage_out[i])
            mod.addConstr(cap[i] <= self.cap_max)

            mod.addConstr(p_storage_in[i] <= self.cap_max - cap_x)
            mod.addConstr(p_storage_out[i] <= cap_x)

            mod.addConstr(p_storage_in[i] <= 200)
            mod.addConstr(p_storage_out[i] <= 200)

            # Diesel and grid bounds
            mod.addConstr(p_diesel[i] <= self.p_diesel_max)
            mod.addConstr(p_grid_in[i] <= 600)

            # Objective function
            obf = (self.c_grid[i] * p_grid_out[i] + self.c_diesel * p_diesel[i] +
                   c_virt[i] * p_storage_in[i] - self.c_grid[i] * p_grid_in[i])
            mod.setObjective(obf)

            feasible = solve(mod)

            # If one of the timestep is not feasible, get out of the loop
            if not feasible:
                break

            models.append(mod)

            # Update the storage capacity
            cap_x = cap[i].X

        return models, feasible

    def _compute_real_cost(self, models):
        """
        Given a list of optimization models, one for each timestep, the method returns the real cost value.
        :param models: list of gurobipy.Model; a list with an optimization model for each timestep.
        :return: float; the real cost of the given solution.
        """

        # Check that the number of optimization models is equal to the number of timestep
        assert len(models) == self.n

        cost = 0
        all_cost = []

        # Compute the total cost considering all the timesteps
        for timestep, model in enumerate(models):
            optimal_p_grid_out = model.getVarByName('p_grid_out_' + str(timestep)).X
            optimal_p_diesel = model.getVarByName('p_diesel_' + str(timestep)).X
            optimal_p_grid_in = model.getVarByName('p_grid_in_' + str(timestep)).X

            cost += (self.c_grid[timestep] * optimal_p_grid_out + self.c_diesel * optimal_p_diesel
                     - self.c_grid[timestep] * optimal_p_grid_in)
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
        else:
            # The reward is the negative real cost
            reward = -self._compute_real_cost(models)

        # The episode has a single timestep
        done = True

        observations = self._get_observations()

        return observations, reward, done, {}

########################################################################################################################


class MarkovianVPPEnv(VPPEnv):
    """
    Gym environment for the Markovian version of the VPP optimization model.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self,
                 predictions,
                 c_grid,
                 shift,
                 noise_std_dev=0.02,
                 savepath=None):
        """
        :param predictions: pandas.Dataframe; predicted PV and Load.
        :param c_grid: numpy.array; c_grid values.
        :param shift: numpy.array; shift values.
        :param noise_std_dev: float; the standard deviation of the additive gaussian noise for the realizations.
        :param savepath: string; if not None, the gurobi models are saved to this directory.
        """

        super(MarkovianVPPEnv, self).__init__(predictions, c_grid, shift, noise_std_dev, savepath)

        # Here we define the observation and action spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.n * 3 + 1,), dtype=np.float32)
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _create_instance_variables(self):
        """
        Create predicted and real, PV and Load for the current instance.
        :return:
        """

        assert self.mr is not None, "Instance index must be initialized"

        super()._create_instance_variables()

        # Set the timestep
        self.timestep = 0

        # Set the cumulative cost
        self.cumulative_cost = 0

    def _get_observations(self):
        """
        Return predicted pv and load values, one-hot encoding of the timestep and the storage.
        :return: numpy.array; pv and load values for the current instance.
        """

        observations = np.concatenate((self.p_ren_pv_pred / np.max(self.p_ren_pv_pred),
                                       self.tot_cons_pred / np.max(self.tot_cons_pred)),
                                      axis=0)
        one_hot_timestep = np.zeros(shape=(self.n, ))
        one_hot_timestep[int(self.timestep)] = 1
        observations = np.concatenate((observations, one_hot_timestep), axis=0)
        observations = np.append(observations, self.storage / self.cap_max)
        observations = np.squeeze(observations)

        return observations

    def _clear(self):
        """
        Clear all the instance dependent variables.
        :return:
        """
        self.p_ren_pv_pred = None
        self.p_ren_pv_real = None
        self.tot_cons_pred = None
        self.tot_cons_real = None
        self.mr = None
        self.storage = self.in_cap
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
        assert self.c_grid is not None, "c_grid must be initialized"
        assert self.shift is not None, "shifts must be initialized before the step function"
        assert self.p_ren_pv_real is not None, "Real PV values must be initialized before the step function"
        assert self.tot_cons_real is not None, "Real Load values must be initialized before the step function"
        assert self.storage is not None, "Storage variable must be initialized"

        # Enforce action space
        c_virt = np.clip(c_virt, self.action_space.low, self.action_space.high)
        c_virt = np.squeeze(c_virt)

        # Create an optimization model
        mod = Model()

        # build variables and define bounds
        p_diesel = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel")
        p_storage_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in")
        p_storage_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out")
        p_grid_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in")
        p_grid_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out")
        cap = mod.addVar(vtype=GRB.CONTINUOUS, name="cap")

        # Shift from Demand Side Energy Management System
        tilde_cons = (self.shift[self.timestep] + self.tot_cons_real[self.timestep])

        # Power balance constraint
        mod.addConstr((self.p_ren_pv_real[self.timestep] + p_storage_out + p_grid_out + p_diesel -
                       p_storage_in - p_grid_in == tilde_cons), "Power_balance")

        # Storage cap
        mod.addConstr(cap == self.storage + p_storage_in - p_storage_out)
        mod.addConstr(cap <= self.cap_max)

        mod.addConstr(p_storage_in <= self.cap_max - self.storage)
        mod.addConstr(p_storage_out <= self.storage)

        mod.addConstr(p_storage_in <= 200)
        mod.addConstr(p_storage_out <= 200)

        # Diesel and grid bounds
        mod.addConstr(p_diesel <= self.p_diesel_max)
        mod.addConstr(p_grid_in <= 600)

        # Objective function
        obf = (self.c_grid[self.timestep] * p_grid_out + self.c_diesel * p_diesel +
               c_virt * p_storage_in - self.c_grid[self.timestep] * p_grid_in)
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

        optimal_p_grid_out = model.getVarByName('p_grid_out').X
        optimal_p_diesel = model.getVarByName('p_diesel').X
        optimal_p_grid_in = model.getVarByName('p_grid_in').X
        optimal_p_storage_in = model.getVarByName('p_storage_in').X
        optimal_p_storage_out = model.getVarByName('p_storage_out').X
        optimal_cap = model.getVarByName('cap').X

        cost = (self.c_grid[self.timestep] * optimal_p_grid_out + self.c_diesel * optimal_p_diesel
                - self.c_grid[self.timestep] * optimal_p_grid_in)

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
        else:
            # The reward is the negative real cost
            reward = -self._compute_real_cost(models)

        # Update the cumulative cost
        self.cumulative_cost -= reward

        observations = self._get_observations()

        # Update the timestep
        self.timestep += 1

        # If we reach the end of the episode or the model is not feasible, we terminate the episode
        if self.timestep == self.n or not feasible:
            done = True
        elif self.timestep < self.n:
            done = False
        else:
            raise Exception(f"Timestep cannot be greater than {self.n}")

        return observations, reward, done, {'feasible': feasible}

########################################################################################################################




