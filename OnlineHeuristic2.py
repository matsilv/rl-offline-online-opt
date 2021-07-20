
from gurobipy import *
import numpy as np
import pandas as pd
import random
import sys


# Optimize VPP planning Model
def solve(mod):
    mod.setParam('OutputFlag',0)
    mod.optimize()
    status = mod.status
    if status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE \
        or status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is infeasible or unbounded')
        exit(1)
                      
    if status != GRB.Status.OPTIMAL:
        print('Optimization was stopped with status %d' % status)
        exit(0)

#greedy heuristic
def heur (mr,namefile):
    
    #timestamp
    n = 96
    
    #price data from GME
    cGrid = np.load('gmePrices.npy')
    cGridS = np.mean(cGrid)
    cGridSt=cGrid
    
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
    
    #if you want to run more than one instance at a time mrT != 1
    for j in range(mrT):
        for i in range(n):

        #create a model
            mod = Model()

        #build variables and define bounds
            pDiesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_"+str(i))
            pStorageIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_"+str(i))
            pStorageOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_"+str(i))
            pGridIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGrid_"+str(i))
            pGridOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGrid_"+str(i))
            cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_"+str(i))
            change[i] = mod.addVar(vtype=GRB.INTEGER, name="change")
            phi[i] = mod.addVar(vtype=GRB.BINARY, name="phi")
            notphi[i] = mod.addVar(vtype=GRB.BINARY, name="notphi")

        #################################################
        #Shift from Demand Side Energy Management System
        #################################################

            tilde_cons[i] = (shift[i]+tot_cons[j][i])

        ####################
        #Model constraints
        ####################

        #more sophisticated storage constraints
            mod.addConstr(notphi[i]==1-phi[i])
            mod.addGenConstrIndicator(phi[i], True, pStorageOut[i], GRB.LESS_EQUAL, 0)
            mod.addGenConstrIndicator(notphi[i], True, pStorageIn[i], GRB.LESS_EQUAL, 0)

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
            mod.addConstr(change[i]>=0)
            mod.addConstr(change[i]>= (phi[i] - phiX))
            mod.addConstr(change[i]>= (phiX - phi[i]))

        #Objective function
            obf = (cGrid[i]*pGridOut[i]+cDiesel*pDiesel[i]-cGridSt[i]*pStorageIn[i]+cGridSt[i]*pStorageOut[i]-cGrid[i]*pGridIn[i]+cGridS*change[i])
        
            mod.setObjective(obf)
            
            
            solve(mod)

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
        for k in range(1, len(objList), 96):
            ob = sum(objList[k:k+96])
        objFinal.append(round(ob,2))
        

        for k in range(1, len(runList), 96):
            run = sum(runList[k:k+96])
        runFinal.append(round(run,2))



        print("\n============================== Solutions of Instance %d  =================================\n\n" %(mr))
        
        print("The solution cost (in keuro) is: %s\n" %(str(np.mean(objFinal))))
        print("The runtime (in sec) is: %s\n" %(str(np.mean(runFinal))))

        print(a4)
        print(a5)
        print(a8)









