"""
    This script contains the original version of the heuristic.
"""

from gurobipy import *
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import sys
from tabulate import tabulate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Union
import warnings
import argparse


########################################################################################################################

cap_max = 1000
in_cap = 800
c_diesel = 0.054
cru = 0.35
p_diesel_max = 1200

########################################################################################################################


# Solve VPP optimization model
def solve(mod):
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


def heur(prices_filename,
         mr=None,
         instances_filename=None,
         p_ren_pv=None,
         tot_cons=None,
         display=False,
         virtual_costs=None):
    """
    Implementation of a simple greedy heuristic. You can load the instances from file and specify the index of the one
    you want to solve or give the instance itself as input.
    :param prices_filename: string; the filepath where the prices are loaded from.
    :param mr: int; index of the instance to be solved.
    :param instances_filename: string; the name of the file from which instances are loaded.
    :param p_ren_pv: numpy.array of shape (n_instances, 96); photovoltaic production at each timestep.
    :param tot_cons: numpy.array of shape (n_instances, 96); electricity demand at each timestep.
    :param display: bool; if True, the solutions is printed to the output.
    :param virtual_costs: np.array; an array with the virtual costs.
    :return: float, list of float, float; final solution cost, list of costs for each timestep and real cost;
                                          None, None and None if the instance can not be solved.
    """

    assert (mr is not None and instances_filename is not None) or (p_ren_pv is not None and tot_cons is not None), \
        "You must specify either the filename from which instances are loaded and the instance index " + \
        "or the instance itself"
    
    # Number of timestamp
    n = 96
    
    # Price data from GME
    assert os.path.isfile(prices_filename), "Prices filename does not exist"
    c_grid = np.load(prices_filename)
    c_grid_s = np.mean(c_grid)

    # Set the virtual costs
    if virtual_costs is None:
        c_virt = c_grid.copy()
    else:
        c_virt = virtual_costs.copy()

    # Capacities, bounds, parameters and prices
    listc = []
    mrT = 1
    objX = np.zeros((mrT, n))
    objList, runList, objFinal, runFinal = [[] for i in range(4)]
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, cap, change, phi, notphi, pDiesel, pStorageIn, pStorageOut, pGridIn, pGridOut, tilde_cons  = [[None]*n for i in range(20)]
    capMax = 1000
    inCap = 800
    capX = inCap
    cDiesel = 0.054
    pDieselMax = 1200
    runtime = 0
    solutions = np.zeros((mrT, n, 9))

    # Optionally, load instances from file
    if instances_filename is not None:
        # read instances
        instances = pd.read_csv(instances_filename)

        # instances pv from file
        instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
        instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))
        pRenPV = [instances['PV(kW)'][mr] for i in range(mrT)]
        np.asarray(pRenPV)

        # instances load from file
        instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
        instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
        tot_cons = [instances['Load(kW)'][mr] for i in range(mrT)]
        np.asarray(tot_cons)

    # Load demand shifts
    shift = np.load('data/optShift.npy')

    # Save all models in a list
    all_models = []
    
    # If you want to run more than one instance at a time mrT != 1
    # FIXME: not supported now
    for j in range(mrT):

        for i in range(n):

            # Create a model
            mod = Model()

            # Build variables and define bounds
            # pDiesel: electricity picked from the diesel power
            pDiesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_"+str(i))
            # pStorageIn: store electricity
            pStorageIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_"+str(i))
            # pStorageOut: pick electricity from the storage and send it to the network
            pStorageOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_"+str(i))
            # pGridIn: electricity to the network (selling)
            pGridIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridIn_"+str(i))
            # pGridOut: buy electricity from the network
            pGridOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridOut_"+str(i))
            # cap: storage capacitance
            cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_"+str(i))

            # change[i] = mod.addVar(vtype=GRB.INTEGER, name="change")
            # phi[i] = mod.addVar(vtype=GRB.BINARY, name="phi")
            # notphi[i] = mod.addVar(vtype=GRB.BINARY, name="notphi")

        #################################################
        # Shift from Demand Side Energy Management System
        #################################################

            tilde_cons[i] = (shift[i]+tot_cons[j][i])

        ####################
        # Model constraints
        ####################

        # More sophisticated storage constraints
            # mod.addConstr(notphi[i]==1-phi[i])
            # mod.addGenConstrIndicator(phi[i], True, pStorageOut[i], GRB.LESS_EQUAL, 0)
            # mod.addGenConstrIndicator(notphi[i], True, pStorageIn[i], GRB.LESS_EQUAL, 0)

        # Power balance constraint
            mod.addConstr((pRenPV[j][i] + pStorageOut[i] + pGridOut[i] + pDiesel[i] - pStorageIn[i] - pGridIn[i] ==
                           tilde_cons[i]),
                          "Power_balance")

            # Storage cap
            mod.addConstr(cap[i] == capX+pStorageIn[i]-pStorageOut[i])
            mod.addConstr(cap[i] <= capMax)

            mod.addConstr(pStorageIn[i] <= capMax-capX)
            mod.addConstr(pStorageOut[i] <= capX)

            mod.addConstr(pStorageIn[i] <= 200)
            mod.addConstr(pStorageOut[i] <= 200)

            # Diesel and Net cap
            mod.addConstr(pDiesel[i] <= pDieselMax)
            mod.addConstr(pGridIn[i] <= 600)
        
            # Storage mode change
            # mod.addConstr(change[i]>=0)
            # mod.addConstr(change[i]>= (phi[i] - phiX))
            # mod.addConstr(change[i]>= (phiX - phi[i]))

            # Objective function
            obf = (cGrid[i]*pGridOut[i]+cDiesel*pDiesel[i]+c_virt[i]*pStorageIn[i]-cGrid[i]*pGridIn[i])
            # for using storage constraints for mode change we have to add cRU*change in the objective function
        
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

            runList.append(mod.Runtime*60)
            runtime += mod.Runtime*60

            # Extract solution values
            a2[i] = pDiesel[i].X
            a4[i] = pStorageIn[i].X
            a5[i] = pStorageOut[i].X
            a3[i] = pRenPV[j][i]
            a6[i] = pGridIn[i].X
            a7[i] = pGridOut[i].X
            a8[i] = cap[i].X
            a1[i] = tilde_cons[i]
            objX[j][i] = mod.objVal
            a9[i] = cGrid[i]
            capX = cap[i].x
            listc.append(capX)
            #phiX = phi[i].x

            solutions[j][i] = [mod.objVal, a3[i], a1[i], capX, a2[i], a4[i], a5[i], a6[i], a7[i]]
        
            objList.append((objX[j][i]))

        a10 = shift

        # Compute the solution cost
        for k in range(0, len(objList), 96):
            ob = sum(objList[k:k+96])
        objFinal.append(ob)

        # Compute the runtime
        for k in range(0, len(runList), 96):
            run = sum(runList[k:k+96])
        runFinal.append(round(run,2))

        real_cost = 0
        diesel_power_consumptions = []
        storage_consumptions = []
        storage_charging = []
        energy_sold = []
        energy_bought = []
        storage_capacity = []
        all_real_costs = []

        # Compute the real cost
        for timestep, model in enumerate(all_models):
            optimal_pGridOut = model.getVarByName('pGridOut_' + str(timestep)).X
            energy_bought.append(optimal_pGridOut)

            optimal_pDiesel = model.getVarByName('pDiesel_' + str(timestep)).X
            diesel_power_consumptions.append(optimal_pDiesel)

            optimal_pGridIn = model.getVarByName('pGridIn_' + str(timestep)).X
            energy_sold.append(optimal_pGridIn)

            optimal_pStorageIn = model.getVarByName('pStorageIn_' + str(timestep)).X
            storage_charging.append(optimal_pStorageIn)

            optimal_pStorageOut = model.getVarByName('pStorageOut_' + str(timestep)).X
            storage_consumptions.append(optimal_pStorageOut)

            optimal_cap = model.getVarByName('cap_' + str(timestep)).X
            storage_capacity.append(optimal_cap)

            cost = (cGrid[timestep] * optimal_pGridOut + cDiesel * optimal_pDiesel -
                          cGrid[timestep] * optimal_pGridIn)
            all_real_costs.append(cost)
            real_cost += cost

        # Optionally, display the solution
        if display:
            print("\n============================== Solutions  =================================\n\n")

            objFinal = np.mean(objFinal)

            print("The solution cost (in keuro) is: %s\n" %(str(np.mean(objFinal))))
            print("The runtime (in sec) is: %s\n" %(str(np.mean(runFinal))))

            timestamps = timestamps_headers(num_timeunits=96)
            table = list()

            visualization_df = pd.DataFrame(index=timestamps)

            visualization_df['Diesel power consumption'] = diesel_power_consumptions.copy()
            diesel_power_consumptions.insert(0, 'pDiesel')
            table.append(diesel_power_consumptions)

            visualization_df['Input to storage'] = storage_charging.copy()
            storage_charging.insert(0, 'pStorageIn')
            table.append(storage_charging)

            visualization_df['Output from storage'] = storage_consumptions
            storage_consumptions.insert(0, 'pStorageOut')
            table.append(storage_consumptions)

            visualization_df['Energy sold'] = energy_sold
            energy_sold.insert(0, 'pGridIn')
            table.append(energy_sold)

            visualization_df['Energy bought'] = energy_bought
            energy_bought.insert(0, 'pGridOut')
            table.append(energy_bought)

            visualization_df['Storage capacity'] = storage_capacity
            storage_capacity.insert(0, 'cap')
            table.append(storage_capacity)

            print(tabulate(table, headers=timestamps, tablefmt='pretty'))

        return {'feasible': True,
                'real cost': real_cost,
                'all real costs': all_real_costs,
                'virtual cost': objFinal,
                'all virtual costs': objList,
                'dataframe': visualization_df}

########################################################################################################################


def solve_optimization_with_virtual_cost(n_timesteps,
                                         cgrid,
                                         shift,
                                         tot_cons,
                                         p_ren_pv,
                                         cvirt):
    """
    Solve the hybrid offline/online optimization problem with additional virtual cost associated to the storage.
    :param n_timesteps: int; the number of timesteps in a day.
    :param cgrid: numpy.array: daily grid prices.
    :param shift: numpy.array; optimal demand shifts.
    :param tot_cons: numpy.array; the total consumption predictions.
    :param p_ren_pv: numpy.array; the photovoltaic predictions.
    :param cvirt: numpy.array; the virtual costs associated to the storage.
    :return:
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
        p_diesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_" + str(i))
        p_storage_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_" + str(i))
        p_storage_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_" + str(i))
        p_grid_in[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridIn_" + str(i))
        p_grid_out[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridOut_" + str(i))
        cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))
        # change[i] = mod.addVar(vtype=GRB.INTEGER, name="change")
        # phi[i] = mod.addVar(vtype=GRB.BINARY, name="phi")
        # notphi[i] = mod.addVar(vtype=GRB.BINARY, name="notphi")

        #################################################
        # Shift from Demand Side Energy Management System
        #################################################

        tilde_cons[i] = (shift[i] + tot_cons[i])

        ####################
        # Model constraints
        ####################

        # more sophisticated storage constraints
        # mod.addConstr(notphi[i]==1-phi[i])
        # mod.addGenConstrIndicator(phi[i], True, pStorageOut[i], GRB.LESS_EQUAL, 0)
        # mod.addGenConstrIndicator(notphi[i], True, pStorageIn[i], GRB.LESS_EQUAL, 0)

        # power balance constraint
        mod.addConstr((p_ren_pv[i] + p_storage_out[i] + p_grid_out[i] + p_diesel[i] - p_storage_in[i] - p_grid_in[i] ==
                       tilde_cons[i]), "Power balance")

        # Storage cap
        mod.addConstr(cap[i] == cap_x + p_storage_in[i] - p_storage_out[i])
        mod.addConstr(cap[i] <= cap_max)

        mod.addConstr(p_storage_in[i] <= cap_max - (cap_x))
        mod.addConstr(p_storage_out[i] <= cap_x)

        mod.addConstr(p_storage_in[i] <= 200)
        mod.addConstr(p_storage_out[i] <= 200)

        # Diesel and Net cap
        mod.addConstr(p_diesel[i] <= p_diesel_max)
        mod.addConstr(p_grid_in[i] <= 600)

        # Storage mode change
        # mod.addConstr(change[i]>=0)
        # mod.addConstr(change[i]>= (phi[i] - phiX))
        # mod.addConstr(change[i]>= (phiX - phi[i]))

        # Objective function
        obf = (cgrid[i] * p_grid_out[i] + c_diesel * p_diesel[i] + cvirt[i] * p_storage_in[i] - cgrid[i] * p_grid_in[i])
        # for using storage constraints for mode change we have to add cRU*change in the objective function

        mod.setObjective(obf)

        assert solve(mod), "The model is infeasible"

        # Compute real and virtual costs
        real_cost += (cgrid[i] * p_grid_out[i].X + c_diesel * p_diesel[i].X - cgrid[i] * p_grid_in[i].X)
        virtual_cost += mod.objVal

        # Update the storage capacity
        cap_x = cap[i].x
        # phiX = phi[i].x

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


def solve_optimization_with_decision_vars(decision_vars,
                                          n_timesteps,
                                          shift,
                                          p_ren_pv,
                                          tot_cons,
                                          cgrid):
    """
    Compute the cost of a complete solution fo the VPP problem.
    :param decision_vars: numpy.array of shape (n_timesteps * 4); the numpy array with the complete solution.
    :param n_timesteps: int; the number of timesteps in a day.
    :param shift: numpy.array; optimal demand shifts.
    :param p_ren_pv: numpy.array; the photovoltaic predictions.
    :param tot_cons: numpy.array; the total consumption predictions.
    :param cgrid: numpy.array; daily grid prices.
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
        obf = (cgrid[i] * grid_out + c_diesel * diesel_power[i] - cgrid[i] * grid_in[i])
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
                      filename: str,
                      virtual_costs: Union[str, np.ndarray] = None,
                      decision_variables: Union[str, np.ndarray] = None,
                      display: bool = False,
                      savepath: str = None):
    """
    Compute the cost either from a complete solution for VPP problem or by solving the hybrid offline/online
    optimization problem with virtual cost associated to the storage.
    :param instance_idx: int; the instance index in the instances file.
    :param filename: string; instances filepath.
    :param virtual_costs: string or numpy.array; the filepath or the numpy array with the virtual costs; it can be None
                                                 if you want to use complete VPP solution.
    :param decision_variables: string or numpy.array; the filepath or the numpy array with the decision variables; it
                                                      can be None if you want to solve the hybrid offline/online
                                                      optimization problem.
    :param display: boolean; True if you want to display the solution, False otherwise.
    :return:
    """

    # Check that either the virtual costs or the decision variables are defined
    assert virtual_costs is not None or decision_variables is not None, \
        "You must specify either the virtual costs or the decision variables"

    # Check that all the required files exist
    gme_prices_filename = os.path.join('data', 'gmePrices.npy')
    opt_shift_filename = os.path.join('data', 'optShift.npy')
    assert os.path.isfile(filename), f"{filename} does not exist"
    assert os.path.isfile(gme_prices_filename), f"{gme_prices_filename} does not exist"
    assert os.path.isfile(opt_shift_filename), f"{opt_shift_filename} does not exist"

    # Create the saving directory if does not exist
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Number of timesteps in a day
    n = 96

    # Price data from GME
    cgrid = np.load(gme_prices_filename)
    cgrids = np.mean(cgrid)

    # read instances
    instances = pd.read_csv(filename)

    # instances pv from file
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))
    p_ren_pv = instances['PV(kW)'][instance_idx]

    # instances load from file
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
    tot_cons = instances['Load(kW)'][instance_idx]

    # Load optimal demand shifts from file
    shift = np.load(opt_shift_filename)

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
                                                 cgrid=cgrid,
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
                                                      cgrid=cgrid)
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
        diesel_power.insert(0, 'pDiesel')
        table.append(diesel_power)

        visualization_df['Input to storage'] = input_storage.copy()
        input_storage.insert(0, 'pStorageIn')
        table.append(input_storage)

        visualization_df['Output from storage'] = output_storage
        output_storage.insert(0, 'pStorageOut')
        table.append(output_storage)

        visualization_df['Energy sold'] = energy_sold
        energy_sold.insert(0, 'pGridIn')
        table.append(energy_sold)

        visualization_df['Energy bought'] = energy_bought
        energy_bought.insert(0, 'pGridOut')
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
    parser.add_argument("solution_filepath", type=str, help="Virtual costs or solution filepath")
    parser.add_argument("mode",
                        type=str,
                        choices=["rl", "hybrid"],
                        help="If 'rl' is selected then you have to directly provide a solution;"
                             + "if 'hybrid' is provided then you have to provide the virtual costs")
    parser.add_argument("instance_id", type=int, help="Instance index in the instances file")
    parser.add_argument("savepath", type=str, help="Where the solution and its cost are saved to")
    parser.add_argument("--display", action="store_true", help="Diplay the solution and cost")

    args = parser.parse_args()

    # Extract the arguments
    instance_filepath = args.instance_filepath
    solutions_filepath = args.solution_filepath
    mode = args.mode
    instance_id = int(args.instance_id)
    savepath = args.savepath
    display = args.display

    # Warn the user on the chosen mode
    warnings.warn(f"You are using {mode} mode")

    # Invoke main function
    if mode == 'rl':
        compute_real_cost(instance_idx=int(instance_id),
                          filename=instance_filepath,
                          decision_variables=solutions_filepath,
                          display=display,
                          savepath=savepath)
    elif mode == 'hybrid':
        compute_real_cost(instance_idx=int(instance_id),
                          filename=instance_filepath,
                          virtual_costs=solutions_filepath,
                          display=display,
                          savepath=savepath)








