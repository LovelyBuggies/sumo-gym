import gym
import sumo_gym
import numpy as np
from sumo_gym.envs.vrp import VRP

vertex_num = 5
depot_num = 1
edge_num = 7
vehicle_num = 3
vertices = np.asarray([(0., 0.), (1., 0.), (2., 1.), (3., 2.), (1., 4.)])
demand = np.asarray([0.] * depot_num + [10.] * (vertex_num - depot_num))
edges = np.asarray([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
departures = np.zeros(vehicle_num).astype(int)
# capacity = np.ones(vertex_num) * 5


env = gym.make('VRP-v0',
    vertex_num=vertex_num,
    depot_num=depot_num,
    edge_num=edge_num,
    vehicle_num=vehicle_num,
    vertices=vertices,
    demand=demand,
    edges=edges,
    departures=departures,
    # capacity=capacity,
)

for i_episode in range(3):
    observation = env.reset()
    for t in range(100):
        # env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
