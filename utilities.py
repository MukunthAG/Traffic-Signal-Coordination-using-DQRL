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

INPUTS = 36
FC1 = 18
FC2 = 18
OUTPUTS = 8
GAMMA = 0.95
ALPHA = 0.01
TARGET_UPDATE = 8

#MEMORY_MANAGER

NUM_EPISODES = 150
CAPACITY = 2000
BATCH_SIZE = 350

#REWARD_PARAMS

MAXP = 12
MAXWT = 200

#ACTION_STRATEGY

EPS_I = 1
EPS_E = 0.001
EPS_DECAY = 0.001
ACTION_DELAY = 10 
TRIGGER_WAITING_STEP = 100

#PERFOMANCE_METER

GRAPH_NAME = "imawss_trigger_dur_loss"
GRAPH_SHOW = False
MAV_COUNT = 25
STARTING_RETURN = -500
STARTING_DUR = 100

#SUMOMANAGEMENT

GUI_ACTIVE = False
TIME_ELAPSE = 0.005
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