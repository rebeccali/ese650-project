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
from symplectic.symoden import SymODEN_T
from experiment_single_embed.data import get_dataset, arrange_data
from symplectic.utils import L2_loss, to_pickle

import time




















