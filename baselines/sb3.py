import numpy as np
import gym
import sumo_gym
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    ChargingStation,
    ElectricVehicles,
)
from DQN.dqn import QNetwork, ReplayBuffer, run_target_update

vertices = [
    Vertex(0.0, 0.0),
    Vertex(0.0, 1.0),
    Vertex(0.0, 2.0),
    Vertex(0.0, 3.0),
    Vertex(0.0, 4.0),
    Vertex(0.0, 5.0),
    Vertex(1.0, 0.0),
    Vertex(1.0, 1.0),
    Vertex(1.0, 2.0),
    Vertex(1.0, 3.0),
    Vertex(1.0, 4.0),
    Vertex(1.0, 5.0),
    Vertex(2.0, 0.0),
    Vertex(2.0, 1.0),
    Vertex(2.0, 2.0),
    Vertex(2.0, 3.0),
    Vertex(2.0, 4.0),
    Vertex(2.0, 5.0),
    Vertex(3.0, 0.0),
    Vertex(3.0, 1.0),
    Vertex(3.0, 2.0),
    Vertex(3.0, 3.0),
    Vertex(3.0, 4.0),
    Vertex(3.0, 5.0),
    Vertex(4.0, 0.0),
    Vertex(4.0, 1.0),
    Vertex(4.0, 2.0),
    Vertex(4.0, 3.0),
    Vertex(4.0, 4.0),
    Vertex(4.0, 5.0),
    Vertex(5.0, 0.0),
    Vertex(5.0, 1.0),
    Vertex(5.0, 2.0),
    Vertex(5.0, 3.0),
    Vertex(5.0, 4.0),
    Vertex(5.0, 5.0),
]

edges = [
    Edge(0, 1),
    Edge(1, 0),
    Edge(1, 2),
    Edge(2, 1),
    Edge(2, 3),
    Edge(3, 2),
    Edge(3, 4),
    Edge(4, 3),
    Edge(4, 5),
    Edge(5, 4),
    Edge(0, 6),
    Edge(6, 0),
    Edge(1, 7),
    Edge(7, 1),
    Edge(2, 8),
    Edge(8, 2),
    Edge(3, 9),
    Edge(9, 3),
    Edge(4, 10),
    Edge(10, 4),
    Edge(5, 11),
    Edge(11, 5),
    Edge(6, 7),
    Edge(7, 6),
    Edge(7, 8),
    Edge(8, 7),
    Edge(8, 9),
    Edge(9, 8),
    Edge(9, 10),
    Edge(10, 9),
    Edge(10, 11),
    Edge(11, 10),
    Edge(6, 12),
    Edge(12, 6),
    Edge(7, 13),
    Edge(13, 7),
    Edge(8, 14),
    Edge(14, 8),
    Edge(9, 15),
    Edge(15, 9),
    Edge(10, 16),
    Edge(16, 10),
    Edge(11, 17),
    Edge(17, 11),
    Edge(12, 13),
    Edge(13, 12),
    Edge(13, 14),
    Edge(14, 13),
    Edge(14, 15),
    Edge(15, 14),
    Edge(15, 16),
    Edge(16, 15),
    Edge(16, 17),
    Edge(17, 16),
    Edge(12, 18),
    Edge(18, 12),
    Edge(13, 19),
    Edge(19, 13),
    Edge(14, 20),
    Edge(20, 14),
    Edge(15, 21),
    Edge(21, 15),
    Edge(16, 22),
    Edge(22, 16),
    Edge(17, 23),
    Edge(23, 17),
    Edge(18, 19),
    Edge(19, 18),
    Edge(19, 20),
    Edge(20, 19),
    Edge(20, 21),
    Edge(21, 20),
    Edge(21, 22),
    Edge(22, 21),
    Edge(22, 23),
    Edge(23, 22),
    Edge(18, 24),
    Edge(24, 18),
    Edge(19, 25),
    Edge(25, 19),
    Edge(20, 26),
    Edge(26, 20),
    Edge(21, 27),
    Edge(27, 21),
    Edge(22, 28),
    Edge(28, 22),
    Edge(23, 29),
    Edge(29, 23),
    Edge(24, 25),
    Edge(25, 24),
    Edge(25, 26),
    Edge(26, 25),
    Edge(26, 27),
    Edge(27, 26),
    Edge(27, 28),
    Edge(28, 27),
    Edge(28, 29),
    Edge(29, 28),
    Edge(24, 30),
    Edge(30, 24),
    Edge(25, 31),
    Edge(31, 25),
    Edge(26, 32),
    Edge(32, 26),
    Edge(27, 33),
    Edge(33, 27),
    Edge(28, 34),
    Edge(34, 28),
    Edge(29, 35),
    Edge(35, 29),
    Edge(30, 31),
    Edge(31, 30),
    Edge(31, 32),
    Edge(32, 31),
    Edge(32, 33),
    Edge(33, 32),
    Edge(33, 34),
    Edge(34, 33),
    Edge(34, 35),
    Edge(35, 34),
]

n_vertex = len(vertices)
n_area = 4
n_edge = len(edges)
n_vehicle = 3
n_electric_vehicle = 3
n_charging_station = 3
electric_vehicles = [ElectricVehicles(i, 1, 220, 50) for i in range(n_electric_vehicle)]
charging_stations = [
    ChargingStation(3, 220, 10),
    ChargingStation(33, 220, 20),
    ChargingStation(22, 220, 30),
]
available_vertices = [
    v
    for v in range(35)
    if v not in (charging_station.location for charging_station in charging_stations)
]

departures = [19, 15, 4]
demands = [
    Demand(6, 4),
    Demand(5, 16),
    Demand(13, 20),
    Demand(28, 11),
    Demand(12, 30),
    Demand(27, 5),
    Demand(1, 28),
    Demand(13, 24),
    Demand(19, 18),
    Demand(11, 32),
]
n_demand = len(demands)
env = gym.make(
    "FMP-v0",
    mode="numerical",
    n_vertex=n_vertex,
    n_area=n_area,
    n_demand=n_demand,
    n_edge=n_edge,
    n_vehicle=n_vehicle,
    n_electric_vehicle=n_electric_vehicle,
    n_charging_station=n_charging_station,
    vertices=vertices,
    demands=demands,
    edges=edges,
    electric_vehicles=electric_vehicles,
    departures=departures,
    charging_stations=charging_stations,
)

"""
DQN
"""
class DQN(object):
    def __init__(
        self,
        agent,
        lr=0.0003,
        batch_size=128,
        max_length=2000,
        tau=100,
        episodes=500,
        initial_size=500,
        gamma=0.95,
        epsilon=0.9,
        decay_rate=0.95,
        min_epsilon=0.01,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.tau = tau
        self.episodes = episodes
        self.initial_size = initial_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

        self.Q_principal = QNetwork(
            env.observation_space(agent).low.size, env.action_space(agent).n, self.lr
        )
        self.Q_target = QNetwork(
            env.observation_space(agent).low.size, env.action_space(agent).n, self.lr
        )
        self.buffer = ReplayBuffer(max_length)

        r_record = []
        total_step = 0
        for episode in range(episodes):
            env.reset()
            done = False
            r_sum = 0

            if episode % 25 == 24:
                epsilon = epsilon * decay_rate
                epsilon = max(min_epsilon, epsilon)

            while not done:
                prob = np.random.rand(1)
                if prob > epsilon:
                    action = self.Q_principal.compute_argmaxQ(np.expand_dims(obs, 0))
                else:
                    action = env.action_space.sample()

                new_obs, r, done, _ = env.step(action)
                done_ = 1 if done else 0

                self.buffer.append((obs, action, r, done_, new_obs))
                while self.buffer.number > max_length:
                    self.buffer.pop()

                if total_step % 10 == 0 and total_step > initial_size:
                    states = []
                    actions = []
                    rewards = []
                    new_states = []

                    samples = self.buffer.sample(batch_size)

                    for j in range(batch_size):
                        states.append(samples[j][0])
                        rewards.append(samples[j][2])
                        new_states.append(samples[j][4])

                    targets = rewards + gamma * self.Q_target.compute_maxQvalues(new_states)

                    for j in range(batch_size):
                        if samples[j][3] == 1:
                            targets[j] = rewards[j]

                    self.Q_principal.train(states, actions, targets)

                if total_step % tau == 0:
                    run_target_update(self.Q_principal, self.Q_target)

                total_step += 1
                r_sum += r
                obs = new_obs

            r_record.append(r_sum)

            fixedWindow = 100
            movingAverage = 0
            if len(r_record) >= fixedWindow:
                movingAverage = np.mean(
                    r_record[len(r_record) - fixedWindow : len(r_record) - 1]
                )
