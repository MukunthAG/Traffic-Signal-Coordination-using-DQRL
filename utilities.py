#IMPORTS

import os
import sys
import math
import time
import torch
import random
import matplotlib
import numpy as np
import traci as tr
import pprint as pp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

#PROCESSING

GPU = torch.device("cuda")

#DQN_ARCHITECTURE      

INPUTS = 24
FC1 = 18
FC2 = 18
OUTPUTS = 8
GAMMA = 0.99
ALPHA = 0.1
TARGET_UPDATE = 10

#MEMORY_MANAGER

NUM_EPISODES = 100
CAPACITY = 10000
BATCH_SIZE = 100

#REWARD_PARAMS

MAXP = 12
MAXWT = 200

#ACTION_STRATEGY

EPS_I = 1
EPS_E = 0.01
EPS_DECAY = 0.0005
ACTION_DELAY = 10 

#SUMOMANAGEMENT

GUI_ACTIVE = False
SUMOCMD = ["sumo-gui" if GUI_ACTIVE else "sumo",
            "-c", "fixedtime.sumocfg",
            "--no-step-log", "true",
            "-W", "true", 
            "--duration-log.disable"]
CONTROLLED_SIGNAL = "TJ2"
NUM_TMS = 12
NUM_OF_ACTIONS = 8
ACTION_PHASES = {
            "0": "GrrGGrGrrGGr",
            "1": "GrrGrGGrrGrG",
            "2": "GGrGrrGGrGrr",
            "3": "GrGGrrGrGGrr",
            "4": "GrrGrrGrrGGG",
            "5": "GrrGGGGrrGrr",
            "6": "GGGGrrGrrGrr",
            "7": "GrrGrrGGGGrr",        
        }
PHASE_INFO = {
            "GrrGGrGrrGGr": "0",
            "GrrGrGGrrGrG": "1",
            "GGrGrrGGrGrr": "2",
            "GrGGrrGrGGrr": "3",
            "GrrGrrGrrGGG": "4",
            "GrrGGGGrrGrr": "5",
            "GGGGrrGrrGrr": "6",
            "GrrGrrGGGGrr": "7",        
        }