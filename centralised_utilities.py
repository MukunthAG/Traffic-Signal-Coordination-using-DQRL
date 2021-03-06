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
from itertools import product

# TRY some different sampling method
# TRY for large networks --> DOING
#PROCESSING

GPU = torch.device("cuda")

#DQN_ARCHITECTURE      

INPUTS = 36
FC1 = 624
FC2 = 624
OUTPUTS = 512
GAMMA = 0.99
ALPHA = 0.001
TARGET_UPDATE = 10

#MEMORY_MANAGER

NUM_EPISODES = 500
CAPACITY = 5000
BATCH_SIZE = 200
MAX_EPISODES = 1200

#REWARD_PARAMS

MAXP = 12
MAXWT = 200

#ACTION_STRATEGY

EPS_I = 1
EPS_E = 0.001
EPS_DECAY = 0.0001
ACTION_DELAY = 10
MAX_PRES = False

#PERFOMANCE_METER

GRAPH_NAME = "centralised_heavy_traffic"
GRAPH_SHOW = False
MAV_COUNT = 50

#SUMOMANAGEMENT

GUI_ACTIVE = False
TIME_ELAPSE = 0.005
SUMOCMD = ["sumo-gui" if GUI_ACTIVE else "sumo",
            "-c", "sumo\\fixedtime.sumocfg",
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