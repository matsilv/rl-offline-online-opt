"""
    This script contains the original version of the heuristic.
"""

from gurobipy import *
import numpy as np
import pandas as pd
import random
import sys
from tabulate import tabulate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


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


# Optimize VPP planning Model
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


# greedy heuristic
def heur(mr=None,
         instances_filename=None,
         pRenPV=None,
         tot_cons=None,
         display=False,
         virtual_costs=None):
    """
    Implementation of a simple heuristic. You can load the instances from file and specify the index of the one you
    want to solve or give the instance itself as input.
    :param mr: int; index of the instance to be solved.
    :param instances_filename: string; the name of the file from which instances are loaded.
    :param pRenPV: numpy.array of shape (n_instances, 96); photovoltaic production at each timestep.
    :param tot_cons: numpy.array of shape (n_instances, 96); electricity demand at each timestep.
    :param display: bool; if True, the solutions is printed to the output.
    :param virtual_costs_filename: string; the name of the file from which the virtual costs are loaded.
    :return: float, list of float, float; final solution cost, list of costs for each timestep and real cost;
                                          None, None and None if the instance can not be solved.
    """

    assert (mr is not None and instances_filename is not None) or (pRenPV is not None and tot_cons is not None), \
        "You must specify either the filename from which instances are loaded and the instance index " + \
        "or the instance itself"
    
    # Number of timestamp
    n = 96
    
    # Price data from GME
    cGrid = np.load('data/gmePrices.npy')
    cGridS = np.mean(cGrid)

    # Set the virtual costs
    if virtual_costs is None:
        c_virt = cGrid.copy()
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


def compute_real_cost(mr, namefile, virtual_costs, display):

    # Check that all the required files exist
    gmePrices_filename = os.path.join('data', 'gmePrices.npy')
    optShift_filename = os.path.join('data', 'optShift.npy')

    assert os.path.isfile(namefile), f"{namefile} does not exist"
    assert os.path.isfile(gmePrices_filename), f"{gmePrices_filename} does not exist"
    assert os.path.isfile(optShift_filename), f"{optShift_filename} does not exist"

    # timestamp
    n = 96

    # price data from GME
    cGrid = np.load(gmePrices_filename)
    cGridS = np.mean(cGrid)

    # Get virtual costs
    if isinstance(virtual_costs, str):
        assert os.path.isfile(virtual_costs), f"{virtual_costs} does not exist"
        cvirt = np.load(virtual_costs, allow_pickle=True)
    elif isinstance(virtual_costs, np.ndarray) and virtual_costs.shape == (n, ):
        cvirt = virtual_costs
    else:
        raise Exception(f"virtual_costs must be a string representing a filepath or a numpy array of shape ({n}, )")

    # capacities, bounds, parameters and prices
    cap, change, phi, notphi, pDiesel, pStorageIn, pStorageOut, pGridIn, pGridOut, tilde_cons = [
        [None] * n for i in range(10)]
    capMax = 1000
    inCap = 800
    capX = inCap
    cDiesel = 0.054
    cRU = 0.35
    pDieselMax = 1200
    runtime = 0
    phiX = 0

    # read instances
    instances = pd.read_csv(namefile)

    # instances pv from file
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))
    pRenPV = instances['PV(kW)'][mr]

    # instances load from file
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
    tot_cons = instances['Load(kW)'][mr]

    # Load optimal demand shifts from file
    shift = np.load(optShift_filename)

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
    for i in range(n):

        # create a model
        mod = Model()

        # build variables and define bounds
        pDiesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_" + str(i))
        pStorageIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_" + str(i))
        pStorageOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_" + str(i))
        pGridIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridIn_" + str(i))
        pGridOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGridOut_" + str(i))
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
        mod.addConstr((pRenPV[i] + pStorageOut[i] + pGridOut[i] + pDiesel[i] - pStorageIn[i] - pGridIn[i] ==
                       tilde_cons[i]), "Power balance")

        # Storage cap
        mod.addConstr(cap[i] == capX + pStorageIn[i] - pStorageOut[i])
        mod.addConstr(cap[i] <= capMax)

        mod.addConstr(pStorageIn[i] <= capMax - (capX))
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
        obf = (cGrid[i] * pGridOut[i] + cDiesel * pDiesel[i] + cvirt[i] * pStorageIn[i] - cGrid[i] * pGridIn[i])
        # for using storage constraints for mode change we have to add cRU*change in the objective function

        mod.setObjective(obf)

        assert solve(mod), "The model is infeasible"

        # Compute real and virtual costs
        real_cost += (cGrid[i] * pGridOut[i].X + cDiesel * pDiesel[i].X - cGrid[i] * pGridIn[i].X)
        virtual_cost += mod.objVal

        # Update the storage capacity
        capX = cap[i].x
        # phiX = phi[i].x

        # Keep track of the decision variables
        energy_bought.append(pGridOut[i].X)
        energy_sold.append(pGridIn[i].X)
        diesel_power_consumptions.append(pDiesel[i].X)
        storage_charging.append(pStorageIn[i].X)
        storage_consumptions.append(pStorageOut[i].X)
        storage_capacity.append(cap[i].X)

        # Keep track of the runtime
        runtime += mod.Runtime * 60

    # Optionally, display the solution and the decision variables
    if display:
        print("\n============================== Solution =================================\n\n")

        print(f'The virtual cost is: {virtual_cost}')
        print(f"The solution cost (in keuro) is: {real_cost}\n")
        print(f"The runtime (in sec) is: {runtime}\n")

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

        # print(tabulate(table, headers=timestamps, tablefmt='pretty'))

        axes = visualization_df.plot(subplots=True, fontsize=12, figsize=(10, 7))
        plt.xlabel('Timestamp', fontsize=14)

        for axis in axes:
            axis.legend(loc=2, prop={'size': 12})
        plt.plot()
        plt.show()

########################################################################################################################


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("ERROR! Usage: python scriptName.py  instancesFilename.csv virtualCostsFilename.npy instanceId\n")
        sys.exit(1)

    nome_script, instances_filename, virtual_costs_filename, instance_idx = sys.argv

    compute_real_cost(mr=int(instance_idx),
                      namefile=instances_filename,
                      virtual_costs=virtual_costs_filename,
                      display=True)








