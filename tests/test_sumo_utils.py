import gym
import sumo_gym
import numpy as np
import random
from sumo_gym.utils.sumo_utils import *
import numpy.typing as npt
from typing import Tuple, Dict
import pytest
from sumo_gym.envs.vrp import VRP

##TODO add tests here instead of in python notebook
def test_sumo_init():
    newSumo = SumoInteractionSingleton(sumo_config="../assets/data/sumo_simulation.sumocfg")
