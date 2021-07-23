#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import sys


def sampler(j):
    np.random.seed(8)
    pRenPV = np.zeros((1000,96))
    pRenPVsamples = np.zeros((1000,96))
    tot_cons = np.zeros((1000,96))
    tot_conssamples = np.zeros((1000,96))
    
    instances = pd.read_csv('InstancesPredictions.csv')
    #instances pv
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))
    for i in range(1000):
        pRenPV[i] = instances['PV(kW)'][i]


    for i in range(1000):
        noise = np.random.normal(0,0.1,96)
        pRenPVsamples[i] = pRenPV[j]+pRenPV[j]*noise

    #instances load
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
    for i in range(1000):
        tot_cons[i] = instances['Load(kW)'][i]


    for i in range(1000):
        noise = np.random.normal(0,0.1,96)
        tot_conssamples[i] = tot_cons[j]+tot_cons[j]*noise

    my_df = pd.DataFrame({'PV(kW)':[np.round(pRenPVsamples[i], decimals=1) for i in range(1000)],'Load(kW)':[np.round(tot_conssamples[i], decimals=1) for i in range(1000)]})

    my_df.to_csv('instancesRealizationsFromPred%d.csv' %(j),index=True, columns=['PV(kW)', 'Load(kW)'])

if __name__ == "__main__":

    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py instancePredictionId\n")
        sys.exit(1)
    nome_script, mr= sys.argv

    sampler(int(mr))
