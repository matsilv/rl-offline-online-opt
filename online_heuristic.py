"""
    Utility functions to invoke the greedy heuristic and evaluate the solutions.
"""

import gurobipy
from gurobipy import GRB
from gurobipy import Model
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
import warnings
import argparse
import os
from utility import instances_preprocessing, timestamps_headers, min_max_scaler


########################################################################################################################

cap_max = 1000
in_cap = 800
c_diesel = 0.054
cru = 0.35
p_diesel_max = 1200

########################################################################################################################


def solve(mod: gurobipy.Model) -> bool:
    """
    Solve an optimization model.
    :param mod: gurobipy.Model; the optimization model to be solved.
    :return: bool; True if the optimal solution is found, False otherwise.
    """

    mod.setParam('OutputFlag', 0)
    mod.optimize()
    status = mod.status
    if status == GRB.Status.UNBOUNDED:
        print('\nThe model is unbounded')
        return False
    elif status == GRB.Status.INFEASIBLE:
        print('\nThe model is infeasible')
        return False
    elif status == GRB.Status.INF_OR_UNBD:
        print('\nThe model is either infeasible or unbounded')
        return False
                      
    if status != GRB.Status.OPTIMAL:
        print('\nOptimization was stopped with status %d' % status)
        return False

    return True

########################################################################################################################


def heur(prices_filename: str,
         shifts_filename: str,
         instances_filename: str = None,
         instance_idx: int = None,
         p_ren_pv: np.array = None,
         tot_cons: np.array = None,
         display: bool = False,
         virtual_costs: np.array = None) -> Tuple[float, List[Union[int, float]]]:
    """
    Implementation of a simple greedy heuristic. You can load the instances from file and specify the index of the one
    you want to solve or give the instance itself as input.
    :param prices_filename: string; the filepath where the prices are loaded from.
    :param shifts_filename: string; the filepath where the shifts are loaded from.
    :param instances_filename: string; the name of the file from which instances are loaded.
    :param instance_idx: int; index of the instance to be solved.
    :param p_ren_pv: numpy.array of shape (n_instances, 96); photovoltaic production at each timestep.
    :param tot_cons: numpy.array of shape (n_instances, 96); electricity demand at each timestep.
    :param display: bool; if True, the solutions is printed to the output.
    :param virtual_costs: np.array; an array with the virtual costs.
    :return: float, list of float, float; final solution cost, list of costs for each timestep and real cost;
                                          None, None and None if the instance can not be solved.
    """

    # Check that the instance index or directly the predictions are provided
    assert (instance_idx is not None and instances_filename is not None) or \
           (p_ren_pv is not None and tot_cons is not None), \
        "You must specify either the filename from which instances are loaded and the instance index " + \
        "or the instance itself"
    
    # Number of timestamp
    n = 96
    
    # Price data from GME
    assert os.path.isfile(prices_filename), "Prices filename does not exist"
    assert os.path.isfile(shifts_filename), "Shifts filename does not exist"
    c_grid = np.load(prices_filename)
    shift = np.load(shifts_filename)

    # Set the virtual costs
    if virtual_costs is None:
        c_virt = c_grid.copy()
    else:
        c_virt = virtual_costs.copy()

    # Capacities, bounds, parameters and prices
    cap_max = 1000
    in_cap = 800
    cap_x = in_cap
    c_diesel = 0.054
    p_diesel_max = 1200
    virtual_cost = 0

    # Optionally, load instances from file
    if instances_filename is not None:
        # Read the instances
        instances = pd.read_csv(instances_filename)
        instances = instances_preprocessing(instances)
        p_ren_pv = instances['PV(kW)'][instance_idx]
        tot_cons = instances['Load(kW)'][instance_idx]

    # Save all models and virtual costs in a list
    all_models = list()
    obj_list = list()

    for i in range(n):

        # Create a model
        mod = Model()

        # Build variables and define bounds
        # p_diesel: electricity picked from the diesel power
        p_diesel = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel_"+str(i))
        # p_storage_in: store electricity
        p_storage_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in_"+str(i))
        # p_storage_out: pick electricity from the storage and send it to the network
        p_storage_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out_"+str(i))
        # p_grid_in: electricity to the network (selling)
        p_grid_in = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in_"+str(i))
        # p_grid_out: buy electricity from the network
        p_grid_out = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out_"+str(i))
        # cap: storage capacitance
        cap = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_"+str(i))

        # Shift from Demand Side Energy Management System
        tilde_cons = (shift[i] + tot_cons[i])

        # Power balance constraint
        mod.addConstr((p_ren_pv[i] + p_storage_out + p_grid_out + p_diesel - p_storage_in - p_grid_in ==
                       tilde_cons), "Power_balance")

        # Storage constraints
        mod.addConstr(cap == cap_x+p_storage_in-p_storage_out)
        mod.addConstr(cap <= cap_max)

        mod.addConstr(p_storage_in <= cap_max-cap_x)
        mod.addConstr(p_storage_out <= cap_x)

        mod.addConstr(p_storage_in <= 200)
        mod.addConstr(p_storage_out <= 200)

        # Diesel and grid bounds
        mod.addConstr(p_diesel <= p_diesel_max)
        mod.addConstr(p_grid_in <= 600)

        # Objective function
        obf = (c_grid[i] * p_grid_out + c_diesel * p_diesel + c_virt[i] * p_storage_in - c_grid[i] * p_grid_in)

        # Solve the gurobi model
        mod.setObjective(obf)
        feasible = solve(mod)

        # If at least one timestep is not feasible then return
        if not feasible:
            return {'feasible': False,
                    'real cost': None,
                    'all real costs': None,
                    'virtual cost': None,
                    'all virtual costs': None}

        # Save all the optimized models in a list
        all_models.append(mod)

        # Update the virtual cost
        virtual_cost += mod.objVal

    # Compute the real cost and the visualize the solution
    real_cost = 0
    diesel_power_consumptions = list()
    storage_consumptions = list()
    storage_charging = list()
    energy_sold = list()
    energy_bought = list()
    storage_capacity = list()
    all_real_costs = list()

    # Compute the real cost
    for timestep, model in enumerate(all_models):
        optimal_p_grid_out = model.getVarByName('p_grid_out_' + str(timestep)).X
        energy_bought.append(optimal_p_grid_out)

        optimal_p_diesel = model.getVarByName('p_diesel_' + str(timestep)).X
        diesel_power_consumptions.append(optimal_p_diesel)

        optimal_p_grid_in = model.getVarByName('p_grid_in_' + str(timestep)).X
        energy_sold.append(optimal_p_grid_in)

        optimal_p_storage_in = model.getVarByName('p_storage_in_' + str(timestep)).X
        storage_charging.append(optimal_p_storage_in)

        optimal_p_storage_out = model.getVarByName('p_storage_out_' + str(timestep)).X
        storage_consumptions.append(optimal_p_storage_out)

        optimal_cap = model.getVarByName('cap_' + str(timestep)).X
        storage_capacity.append(optimal_cap)

        cost = (c_grid[timestep] * optimal_p_grid_out + c_diesel * optimal_p_diesel -
                      c_grid[timestep] * optimal_p_grid_in)
        all_real_costs.append(cost)
        real_cost += cost

    # Solution dataframe
    visualization_df = None

    # Optionally, display the solution
    if display:

        print(f"The solution cost (in keuro) is: {real_cost}\n")

        timestamps = timestamps_headers(num_timeunits=n)
        table = list()

        visualization_df = pd.DataFrame(index=timestamps)

        visualization_df['Diesel power consumption'] = diesel_power_consumptions.copy()
        diesel_power_consumptions.insert(0, 'p_diesel')
        table.append(diesel_power_consumptions)

        visualization_df['Input to storage'] = storage_charging.copy()
        storage_charging.insert(0, 'p_storage_in')
        table.append(storage_charging)

        visualization_df['Output from storage'] = storage_consumptions
        storage_consumptions.insert(0, 'p_storage_out')
        table.append(storage_consumptions)

        visualization_df['Energy sold'] = energy_sold
        energy_sold.insert(0, 'p_grid_in')
        table.append(energy_sold)

        visualization_df['Energy bought'] = energy_bought
        energy_bought.insert(0, 'p_grid_out')
        table.append(energy_bought)

        visualization_df['Storage capacity'] = storage_capacity
        storage_capacity.insert(0, 'cap')
        table.append(storage_capacity)

        print(tabulate(table, headers=timestamps, tablefmt='pretty'))

    return {'feasible': True,
            'real cost': real_cost,
            'all real costs': all_real_costs,
            'virtual cost': virtual_cost,
            'all virtual costs': obj_list,
            'dataframe': visualization_df}

########################################################################################################################


def solve_optimization_with_virtual_cost(n_timesteps: int,
                                         c_grid: np.array,
                                         shift: np.array,
                                         tot_cons: np.array,
                                         p_ren_pv: np.array,
                                         cvirt: np.array) -> Tuple[dict, float, float]:
    """
    Solve the hybrid offline/online optimization problem with additional virtual cost associated to the storage.
    :param n_timesteps: int; the number of timesteps in a day.
    :param c_grid: numpy.array: daily grid prices.
    :param shift: numpy.array; optimal demand shifts.
    :param tot_cons: numpy.array; the total consumption predictions.
    :param p_ren_pv: numpy.array; the photovoltaic predictions.
    :param cvirt: numpy.array; the virtual costs associated to the storage.
    :return: dict, float, float; a dictionary with the decision variables, the real and virtual costs.
    """

    # Check that the virtual costs have length (n_timesteps, )
    assert cvirt.shape == (n_timesteps, ), f"cvirt must have shape ({n_timesteps}, )"

    # Capacities, bounds, parameters and prices
    cap, change, phi, notphi, p_diesel, p_storage_in, p_storage_out, p_grid_in, p_grid_out, tilde_cons = [
        [None] * n_timesteps for _ in range(10)]

    cap_x = in_cap
    runtime = 0
    phi_x = 0

    # Utility variables to keep track of the results
    real_cost = 0
    virtual_cost = 0
    diesel_power_consumptions = []
    storage_consumptions = []
    storage_charging = []
    energy_sold = []
    energy_bought = []
    storage_capacity = []

    # Solve each optimization step
    for i in range(n_timesteps):
        # create a model
        mod = Model()

        # build variables and define bounds
        p_diesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_diesel_" + str(i))
        p_storage_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_in_" + str(i))
        p_storage_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_storage_out_" + str(i))
        p_grid_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_in_" + str(i))
        p_grid_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="p_grid_out_" + str(i))
        cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))

        # Shift from Demand Side Energy Management System
        tilde_cons[i] = (shift[i] + tot_cons[i])

        # Power balance constraint
        mod.addConstr((p_ren_pv[i] + p_storage_out[i] + p_grid_out[i] + p_diesel[i] - p_storage_in[i] - p_grid_in[i] ==
                       tilde_cons[i]), "Power balance")

        # Storage cap
        mod.addConstr(cap[i] == cap_x + p_storage_in[i] - p_storage_out[i])
        mod.addConstr(cap[i] <= cap_max)

        mod.addConstr(p_storage_in[i] <= cap_max - cap_x)
        mod.addConstr(p_storage_out[i] <= cap_x)

        mod.addConstr(p_storage_in[i] <= 200)
        mod.addConstr(p_storage_out[i] <= 200)

        # Diesel and Net cap
        mod.addConstr(p_diesel[i] <= p_diesel_max)
        mod.addConstr(p_grid_in[i] <= 600)

        # Objective function
        obf = (c_grid[i] * p_grid_out[i] + c_diesel * p_diesel[i] + cvirt[i] * p_storage_in[i] - c_grid[i] * p_grid_in[i])
        mod.setObjective(obf)

        # Check that the model is feasible
        assert solve(mod), "The model is infeasible"

        # Compute real and virtual costs
        real_cost += (c_grid[i] * p_grid_out[i].X + c_diesel * p_diesel[i].X - c_grid[i] * p_grid_in[i].X)
        virtual_cost += mod.objVal

        # Update the storage capacity
        cap_x = cap[i].x

        # Keep track of the decision variables
        energy_bought.append(p_grid_out[i].X)
        energy_sold.append(p_grid_in[i].X)
        diesel_power_consumptions.append(p_diesel[i].X)
        storage_charging.append(p_storage_in[i].X)
        storage_consumptions.append(p_storage_out[i].X)
        storage_capacity.append(cap[i].X)

        # Keep track of the runtime
        runtime += mod.Runtime * 60

    # Create a dictionary with the decision variables
    decision_vars = dict()
    decision_vars['Energy bought'] = energy_bought
    decision_vars['Energy sold'] = energy_sold
    decision_vars['Diesel power'] = diesel_power_consumptions
    decision_vars['Input storage'] = storage_charging
    decision_vars['Output storage'] = storage_consumptions
    decision_vars['Storage capacity'] = storage_capacity

    return decision_vars, real_cost, virtual_cost

########################################################################################################################


def solve_optimization_with_decision_vars(decision_vars: np.array,
                                          n_timesteps: int,
                                          shift: np.array,
                                          p_ren_pv: np.array,
                                          tot_cons: np.array,
                                          c_grid: np.array) -> Tuple[float, dict]:
    """
    Compute the cost of a complete solution fo the VPP problem.
    :param decision_vars: numpy.array of shape (n_timesteps * 4); the numpy array with the complete solution.
    :param n_timesteps: int; the number of timesteps in a day.
    :param shift: numpy.array; optimal demand shifts.
    :param p_ren_pv: numpy.array; the photovoltaic predictions.
    :param tot_cons: numpy.array; the total consumption predictions.
    :param c_grid: numpy.array; daily grid prices.
    :return: float and dict; the real cost value and the dictionary with the solution.
    """

    # We follow this convention:
    # action[0] -> input to storage
    # action[1] -> output from storage
    # action[2] -> power sold to the grid
    # action[3] -> power generated from the diesel source

    # Check that the decision variables have length (n_timesteps, 4)
    assert decision_vars.shape == (n_timesteps, 4), f"decision_vars must have shape ({n_timesteps}, 4)"

    storage_in = decision_vars[:, 0]
    storage_out = decision_vars[:, 1]
    grid_in = decision_vars[:, 2]
    diesel_power = decision_vars[:, 3]
    all_grid_out = []

    # Check that decision variables are n_timesteps long
    assert storage_in.shape == (n_timesteps,)
    assert storage_out.shape == (n_timesteps,)
    assert grid_in.shape == (n_timesteps,)
    assert diesel_power.shape == (n_timesteps,)

    # Rescale the actions in their feasible ranges
    storage_in = min_max_scaler(starting_range=(-1, 1), new_range=(0, 200), value=storage_in)
    storage_out = min_max_scaler(starting_range=(-1, 1), new_range=(0, 200), value=storage_out)
    grid_in = min_max_scaler(starting_range=(-1, 1), new_range=(0, 600), value=grid_in)
    diesel_power = min_max_scaler(starting_range=(-1, 1), new_range=(0, p_diesel_max), value=diesel_power)

    # Initialize the storage capacity and a list to keep track of step-by-step capacity
    cap_x = in_cap
    capacity = []

    # Keep track of the cumulative cost
    cost = 0

    # Loop for each timestep
    for i in range(n_timesteps):

        # Shift from Demand Side Energy Management System
        tilde_cons = (shift[i] + tot_cons[i])

        # Set the power out from the grid so that the power balance constraint is satisfied
        grid_out = tilde_cons - p_ren_pv[i] - storage_out[i] - diesel_power[i] + storage_in[i] + grid_in[i]
        all_grid_out.append(grid_out)

        # Compute the cost
        obf = (c_grid[i] * grid_out + c_diesel * diesel_power[i] - c_grid[i] * grid_in[i])
        cost += obf

        # If the storage constraints are not satisfied then the solution is not feasible
        if storage_in[i] > cap_max - cap_x or storage_out[i] > cap_x:
            cost = np.inf
            break

        # Update the storage capacity
        old_cap_x = cap_x
        cap_x = cap_x + storage_in[i] - storage_out[i]
        capacity.append(cap_x)

        # Check that the constraints are satisfied
        assert cap_x == old_cap_x + storage_in[i] - storage_out[i]
        assert cap_x <= cap_max
        assert storage_in[i] <= cap_max - old_cap_x
        assert storage_out[i] <= old_cap_x
        assert storage_in[i] <= 200
        assert storage_out[i] <= 200
        assert diesel_power[i] <= p_diesel_max
        assert grid_in[i] <= 600
        power_balance = p_ren_pv[i] + storage_out[i] + grid_out + diesel_power[i] - storage_in[i] - grid_in[i]
        assert_almost_equal(power_balance, tilde_cons, decimal=10)

    # Create a dictionary with the decision variables
    decision_vars = dict()
    decision_vars['Energy bought'] = all_grid_out
    decision_vars['Energy sold'] = list(grid_in)
    decision_vars['Diesel power'] = list(diesel_power)
    decision_vars['Input storage'] = list(storage_in)
    decision_vars['Output storage'] = list(storage_out)
    decision_vars['Storage capacity'] = capacity

    return cost, decision_vars

########################################################################################################################


def compute_real_cost(instance_idx: int,
                      predictions_filepath: str,
                      shifts_filepath: str,
                      prices_filepath: str,
                      virtual_costs: Union[str, np.ndarray] = None,
                      decision_variables: Union[str, np.ndarray] = None,
                      display: bool = False,
                      savepath: str = None):
    """
    Compute the cost either from a complete solution for VPP problem or by solving the hybrid offline/online
    optimization problem with virtual cost associated to the storage.
    :param instance_idx: int; the instance index in the instances file.
    :param predictions_filepath: string; where instances are loaded from.
    :param shifts_filepath: string; where optimal shifts are loaded from.
    :param prices_filepath: string; where prices are loaded from.
    :param virtual_costs: string or numpy.array; the filepath or the numpy array with the virtual costs; it can be None
                                                 if you want to use complete VPP solution.
    :param decision_variables: string or numpy.array; the filepath or the numpy array with the decision variables; it
                                                      can be None if you want to solve the hybrid offline/online
                                                      optimization problem.
    :param display: boolean; True if you want to display the solution, False otherwise.
    :param savepath: string; where the real cost and the solution are saved to.
    :return:
    """

    # Check that either the virtual costs or the decision variables are defined
    assert virtual_costs is not None or decision_variables is not None, \
        "You must specify either the virtual costs or the decision variables"

    # Check that all the required files exist
    assert os.path.isfile(predictions_filepath), f"{predictions_filepath} does not exist"
    assert os.path.isfile(prices_filepath), f"{prices_filepath} does not exist"
    assert os.path.isfile(shifts_filepath), f"{shifts_filepath} does not exist"

    # Create the saving directory if does not exist
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Number of timesteps in a day
    n = 96

    # Price data from GME
    c_grid = np.load(prices_filepath)

    # Read the instances
    instances = pd.read_csv(predictions_filepath)
    instances = instances_preprocessing(instances)
    p_ren_pv = instances['PV(kW)'][instance_idx]
    tot_cons = instances['Load(kW)'][instance_idx]

    # Load optimal demand shifts from file
    shift = np.load(shifts_filepath)

    # Compute the cost of the solution...

    # ...from virtual costs or...
    if virtual_costs is not None:
        # Get virtual costs
        if isinstance(virtual_costs, str):
            assert os.path.isfile(virtual_costs), f"{virtual_costs} does not exist"
            cvirt = np.load(virtual_costs, allow_pickle=True)
        elif isinstance(virtual_costs, np.ndarray) and virtual_costs.shape == (n, ):
            cvirt = virtual_costs
        else:
            raise Exception(f"virtual_costs must be a string representing a filepath or a numpy array of shape ({n}, )")

        solution, real_cost, virtual_cost = \
            solve_optimization_with_virtual_cost(n_timesteps=n,
                                                 c_grid=c_grid,
                                                 shift=shift,
                                                 p_ren_pv=p_ren_pv,
                                                 tot_cons=tot_cons,
                                                 cvirt=cvirt)
    # ...or from a complete solution
    elif decision_variables is not None:
        # Get decision variables
        if isinstance(decision_variables, str):
            assert os.path.isfile(decision_variables), f"{decision_variables} does not exist"
            decision_vars = np.load(decision_variables, allow_pickle=True)
        elif isinstance(decision_variables, np.ndarray) and virtual_costs.shape == (n * 4):
            decision_vars = decision_variables
        else:
            raise Exception(f"decision_variables must be a string representing a filepath or" + \
                            f"a numpy array of shape ({n * 4}, )")

        # The virtual cost has no meaning when the complete solution is provided
        virtual_cost = -np.inf
        warnings.warn('The virtual cost has no meaning when the complete solution is provided')

        if np.array_equal(decision_vars, np.inf):
            real_cost = np.inf
            solution = None
        else:
            real_cost, solution = \
                solve_optimization_with_decision_vars(decision_vars=decision_vars,
                                                      n_timesteps=n,
                                                      shift=shift,
                                                      p_ren_pv=p_ren_pv,
                                                      tot_cons=tot_cons,
                                                      c_grid=c_grid)
        if real_cost == np.inf:
            print('Found infeasible solution')

    # Print the virtual and real costs
    print("\n============================== Solution =================================\n\n")

    print(f'The virtual cost is: {virtual_cost}')
    print(f"The solution cost (in keuro) is: {real_cost}\n")

    # Check that the solution found is feasible
    if real_cost != np.inf:

        # Get the decision variables
        energy_bought = solution['Energy bought']
        energy_sold = solution['Energy sold']
        diesel_power = solution['Diesel power']
        input_storage = solution['Input storage']
        output_storage = solution['Output storage']
        storage_capacity = solution['Storage capacity']

        # Create a dataframe with all the solution variables
        timestamps = timestamps_headers(num_timeunits=96)
        table = list()

        visualization_df = pd.DataFrame(index=timestamps)

        visualization_df['Diesel power consumption'] = diesel_power.copy()
        diesel_power.insert(0, 'p_diesel')
        table.append(diesel_power)

        visualization_df['Input to storage'] = input_storage.copy()
        input_storage.insert(0, 'p_storage_in')
        table.append(input_storage)

        visualization_df['Output from storage'] = output_storage
        output_storage.insert(0, 'p_storage_out')
        table.append(output_storage)

        visualization_df['Energy sold'] = energy_sold
        energy_sold.insert(0, 'p_grid_in')
        table.append(energy_sold)

        visualization_df['Energy bought'] = energy_bought
        energy_bought.insert(0, 'p_grid_out')
        table.append(energy_bought)

        visualization_df['Storage capacity'] = storage_capacity
        storage_capacity.insert(0, 'cap')
        table.append(storage_capacity)

        # Optionally, display the decision variables
        if display:
            # print(tabulate(table, headers=timestamps, tablefmt='pretty'))

            axes = visualization_df.plot(subplots=True, fontsize=12, figsize=(10, 7))
            plt.xlabel('Timestamp', fontsize=14)

            for axis in axes:
                axis.legend(loc=2, prop={'size': 12})
            plt.plot()
            plt.show()

        # Optionally, save the solution and the cost
        if savepath is not None:
            visualization_df.to_csv(os.path.join(savepath, f'{instance_idx}_solution.csv'))
    np.save(os.path.join(savepath, f'{instance_idx}_cost.npy'), real_cost)

########################################################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("instance_filepath", type=str, help="User demand and RES productions forecasting filepath")
    parser.add_argument("shifts_filepath", type=str, help="Optimal shifts filepath")
    parser.add_argument("prices_filepath", type=str, help="Prices filepath")
    parser.add_argument("solution_filepath", type=str, help="Virtual costs or solution filepath")
    parser.add_argument("mode",
                        type=str,
                        choices=["rl", "hybrid"],
                        help="If 'rl' is selected then you have to directly provide a solution;"
                             + "if 'hybrid' is provided then you have to provide the virtual costs")
    parser.add_argument("instance_id", type=int, help="Instance index in the instances file")
    parser.add_argument("savepath", type=str, help="Where the solution and its cost are saved to")
    parser.add_argument("--display", action="store_true", help="Display the solution and cost")

    args = parser.parse_args()

    # Extract the arguments
    instance_filepath = args.instance_filepath
    solutions_filepath = args.solution_filepath
    shifts_filepath = args.shifts_filepath
    prices_filepath = args.prices_filepath
    mode = args.mode
    instance_id = int(args.instance_id)
    savepath = args.savepath
    display = args.display

    # Invoke main function
    if mode == 'rl':
        compute_real_cost(instance_idx=int(instance_id),
                          predictions_filepath=instance_filepath,
                          shifts_filepath=shifts_filepath,
                          prices_filepath=prices_filepath,
                          decision_variables=solutions_filepath,
                          display=display,
                          savepath=savepath)
    elif mode == 'hybrid':
        compute_real_cost(instance_idx=int(instance_id),
                          predictions_filepath=instance_filepath,
                          shifts_filepath=shifts_filepath,
                          prices_filepath=prices_filepath,
                          virtual_costs=solutions_filepath,
                          display=display,
                          savepath=savepath)








