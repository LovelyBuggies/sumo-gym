import numpy as np
import gym
import sumo_gym
from sumo_gym.envs.vrp import VRP
import matplotlib
import matplotlib.pyplot as plt

env = gym.make(
    "FMP-v0",
    net_xml_file_path="assets/data/jumbo.net.xml",
    demand_xml_file_path="assets/data/jumbo.rou.xml",
    additional_xml_file_path="assets/data/jumbo_charging_station_additional.xml",
)

env.render()
for i_episode in range(1):
    observation = env.reset()
    for t in range(150):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps.\n".format(t + 1))
            break

env.close()
