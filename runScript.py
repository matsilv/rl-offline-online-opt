#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from gurobipy import *
import numpy as np
import pandas as pd
import random
import sys
import OnlineHeuristic2 as onHeur


if __name__ == '__main__':
    
    if(len(sys.argv)<2):
        print("ERROR! Usage: python scriptName.py  filename.csv instanceId\n")
        sys.exit(1)
    nome_script, fileName, mr= sys.argv

    mr = int(mr)
    onHeur.heur(mr,fileName)
