#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 01:01:54 2020

@author: walker
"""


import torch, argparse
import numpy as np

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from symplectic.nn_models import MLP, PSD
from symplectic.symoden import SymODEN_Q
#from experiment_single_embed.data import get_dataset, arrange_data
from symplectic.utils import L2_loss, to_pickle

import time

device = torch.device('cpu')
M_net = PSD(4, 300, 3).to(device) #(input dim, hidden dim, diagonal dim of output matrix)
g_net = MLP(4, 200, 2).to(device)
V_net = MLP(4, 50, 1).to(device)
model = SymODEN_Q(M_net=M_net, V_net=V_net, g_net=g_net, device=device, baseline=False, structure=True).to(device)

# test SymODEN_Q - x,y,dx,dy,costh,sinth,dth,u1,u2]
x = torch.tensor([[1,2,0.1,0.2,0,1,0.1,1,1]])
model.forward(0,x)












