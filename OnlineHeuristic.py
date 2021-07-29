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


# Optimize VPP planning Model
def solve(mod):
    """
    Solve an optimization model.
    :param mod: gurobipy.Model; the optimization model to be solved.
    :return: bool; True if the optimal solution is found, False otherwise.
    """

    mod.setParam('OutputFlag',0)
    mod.optimize()
    status = mod.status
    if status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE \
        or status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is infeasible or unbounded')
        return False
                      
    if status != GRB.Status.OPTIMAL:
        print('Optimization was stopped with status %d' % status)
        return False

    return True

#greedy heuristic
def heur (mr=None, namefile=None, pRenPV=None, tot_cons=None):
    """
    Implementation of a simple heuristic.
    :param mr: int; index of the instance to be solved.
    :param namefile: string; the name of the file from which instances are loaded.
    :param pRenPV: numpy.array of shape (n_instances, 96); photovoltaic production at each timestep.
    :param tot_cons: numpy.array of shape (n_instances, 96); electricity demand at each timestep.
    :return: float, list of float; final solution cost and list of costs for each timestep; None and None if the
                                   instance can not be solved.
    """

    assert (mr is not None and namefile is not None) or (pRenPV is not None and tot_cons is not None), \
        "You must specify either the filename from which instances are loaded or the instance itself"
    
    #timestamp
    n = 96
    
    #price data from GME
    cGrid = np.load('gmePrices.npy')
    cGridS = np.mean(cGrid)
    
    #capacities, bounds, parameters and prices
    listc = []
    mrT = 1
    objX = np.zeros((mrT,n))
    objTot= [None]*mrT
    objList, runList, objFinal, runFinal = [[] for i in range(4)]
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, cap, change, phi, notphi, pDiesel, pStorageIn, pStorageOut, pGridIn, pGridOut, tilde_cons  = [[None]*n for i in range(20)]
    capMax = 1000
    inCap = 800
    capX = inCap
    cDiesel = 0.054
    cRU = 0.35
    pDieselMax = 1200
    runtime = 0
    phiX = 0
    solutions = np.zeros((mrT,n,9))

    if namefile is not None:
        #read instances
        instances = pd.read_csv(namefile)

        #instances pv from file
        instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
        instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))
        pRenPV = [instances['PV(kW)'][mr] for i in range(mrT)]
        np.asarray(pRenPV)

        #instances load from file
        instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
        instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
        tot_cons = [instances['Load(kW)'][mr] for i in range(mrT)]
        np.asarray(tot_cons)
    
    shift = np.load('optShift.npy')

    all_models = []
    
    #if you want to run more than one instance at a time mrT != 1
    for j in range(mrT):
        for i in range(n):

        #create a model
            mod = Model()

        #build variables and define bounds
            # pDiesel: electricity picked from the diesel power
            pDiesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_"+str(i))
            # GridOut (acquisto)
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
            #change[i] = mod.addVar(vtype=GRB.INTEGER, name="change")
            #phi[i] = mod.addVar(vtype=GRB.BINARY, name="phi")
            #notphi[i] = mod.addVar(vtype=GRB.BINARY, name="notphi")

        #################################################
        #Shift from Demand Side Energy Management System
        #################################################

            tilde_cons[i] = (shift[i]+tot_cons[j][i])

        ####################
        #Model constraints
        ####################

        #more sophisticated storage constraints
            #mod.addConstr(notphi[i]==1-phi[i])
            #mod.addGenConstrIndicator(phi[i], True, pStorageOut[i], GRB.LESS_EQUAL, 0)
            #mod.addGenConstrIndicator(notphi[i], True, pStorageIn[i], GRB.LESS_EQUAL, 0)

        #power balance constraint
            mod.addConstr((pRenPV[j][i]+pStorageOut[i]+pGridOut[i]+pDiesel[i]-pStorageIn[i]-pGridIn[i] == tilde_cons[i]), "Power balance")

        #Storage cap
            mod.addConstr(cap[i]==capX+pStorageIn[i]-pStorageOut[i])
            mod.addConstr(cap[i]<=capMax)

            mod.addConstr(pStorageIn[i]<=capMax-(capX))
            mod.addConstr(pStorageOut[i]<=capX)

            mod.addConstr(pStorageIn[i]<=200)
            mod.addConstr(pStorageOut[i]<=200)


        #Diesel and Net cap
            mod.addConstr(pDiesel[i]<=pDieselMax)
            mod.addConstr(pGridIn[i]<=600)
        
        #Storage mode change
            #mod.addConstr(change[i]>=0)
            #mod.addConstr(change[i]>= (phi[i] - phiX))
            #mod.addConstr(change[i]>= (phiX - phi[i]))

        #Objective function
            obf = (cGrid[i]*pGridOut[i]+cDiesel*pDiesel[i]+cGrid[i]*pStorageIn[i]-cGrid[i]*pGridIn[i])
            #for using storage constraints for mode change we have to add cRU*change in the objective function
        
            mod.setObjective(obf)

            mod.write('model.lp')

            feasible = solve(mod)

            if not feasible:
                return None, None

            all_models.append(mod)

            runList.append(mod.Runtime*60)
            runtime += mod.Runtime*60

            #extract x values
            a2[i] = pDiesel[i].X
            a4[i]  = pStorageIn[i].X
            a5[i]  = pStorageOut[i].X
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
        data = np.array([a1, a2, a3, a9, a6, a7, a4, a5, a8, a10])
        for k in range(0, len(objList), 96):
            ob = sum(objList[k:k+96])
        objFinal.append(ob)
        

        for k in range(0, len(runList), 96):
            run = sum(runList[k:k+96])
        runFinal.append(round(run,2))

        print("\n============================== Solutions of Instance %d  =================================\n\n" %(mr))

        objFinal = np.mean(objFinal)
        
        print("The solution cost (in keuro) is: %s\n" %(str(np.mean(objFinal))))
        print("The runtime (in sec) is: %s\n" %(str(np.mean(runFinal))))

        timestamps = timestamps_headers(num_timeunits=96)
        table = list()

        diesel_power_consumptions = a2.copy()
        diesel_power_consumptions.insert(0, 'pDiesel')
        table.append(diesel_power_consumptions)

        storage_consumptions = a4.copy()
        storage_consumptions.insert(0, 'pStorageIn')
        table.append(storage_consumptions)

        storage_charging = a5.copy()
        storage_charging.insert(0, 'pStorageOut')
        table.append(storage_charging)

        energy_sold = a6.copy()
        energy_sold.insert(0, 'pGridIn')
        table.append(energy_sold)

        energy_bought = a7.copy()
        energy_bought.insert(0, 'pGridOut')
        table.append(energy_bought)

        storage_capacity = a8.copy()
        storage_capacity.insert(0, 'cap')
        table.append(storage_capacity)

        all_costs = list(objX[0])
        all_costs.insert(0, 'Cost')
        table.append(all_costs)

        print(tabulate(table, headers=timestamps, tablefmt='pretty'))

        return objFinal, objList

########################################################################################################################










