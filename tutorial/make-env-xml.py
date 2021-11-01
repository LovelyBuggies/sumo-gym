import numpy as np
import gym
import sumo_gym
from sumo_gym.envs.vrp import VRP
import matplotlib
import matplotlib.pyplot as plt

env = gym.make(
    "VRP-v0",
    net_xml_file_path="assets/data/network.net.xml",
    demand_xml_file_path="assets/data/demand.rou.xml",
)
env.render()
plt.savefig("./img/env_init.pdf")

for i_episode in range(3):
    observation = env.reset()
    for t in range(10):
        env.render()
        plt.savefig(f"./img/env_{i_episode}_{t}.pdf")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(info)
        if done:
            print("Episode finished after {} timesteps.\n".format(t + 1))
            break

env.close()
