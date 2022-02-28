import numpy as np
import json
import gym
import sumo_gym
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    ChargingStation,
    ElectricVehicles,
    get_safe_indicator,
)
from DQN.dqn import QNetwork, ReplayBuffer, run_target_update
from statistics import mean

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
electric_vehicles = [ElectricVehicles(i, 1, 220, 35) for i in range(n_electric_vehicle)]
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
    verbose=1,
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


class MADQN(object):
    def __init__(
        self,
        env,
        lr=0.003,
        batch_size=4,
        tau=50,
        episodes=2000,
        gamma=0.95,
        epsilon=1.0,
        decay_period=15,
        decay_rate=0.95,
        min_epsilon=0.01,
        initial_step=50,
    ):
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.initial_step = initial_step

        self.q_principal = {
            agent: QNetwork(
                1,
                env.action_space(agent).n,
                self.lr,
            )
            for agent in self.env.agents
        }
        self.q_target = {
            agent: QNetwork(
                1,
                env.action_space(agent).n,
                self.lr,
            )
            for agent in self.env.agents
        }
        self.replay_buffer = {
            agent: ReplayBuffer() for agent in self.env.possible_agents
        }
        self.total_step = {agent: 0 for agent in self.env.possible_agents}

    def train(self):
        reward_record = {}
        loss_mean_record = {}
        for episode in range(self.episodes):
            env.reset()
            episode_step = {agent: 0 for agent in env.possible_agents}
            reward_sum = {agent: 0 for agent in env.possible_agents}
            loss_in_episode = {agent: list() for agent in env.possible_agents}
            if episode % self.decay_period == 0:
                self.epsilon *= self.decay_rate
                self.epsilon = max(self.min_epsilon, self.epsilon)

            prev_action = {agent: None for agent in env.possible_agents}
            for agent in env.agent_iter():
                observation, reward, done, info = env.last()
                if observation != 3 and prev_action[agent] is not None and prev_action[agent] != 2:
                    if self.replay_buffer[agent] and episode_step != 0:
                        self.replay_buffer[agent][-1][2] = observation

                    self.replay_buffer[agent].push(
                        [observation, prev_action[agent], None, reward]
                    )

                if observation == 3:
                    action = 2
                elif np.random.rand(1) < self.epsilon:
                    action = env.action_space(agent).sample()
                else:
                    action = self.q_principal[agent].compute_argmax_q(observation)

                env.step(action)
                prev_action[agent] = action
                episode_step[agent] += 1

                if done:
                    agent_idx = env.agent_name_idx_mapping[agent]
                    self.replay_buffer[agent][-1][2] = get_safe_indicator(
                        env.fmp.vertices,
                        env.fmp.edges,
                        env.fmp.demands,
                        env.fmp.charging_stations,
                        env.fmp.electric_vehicles[agent_idx].location,
                        env.fmp.electric_vehicles[agent_idx].battery,
                    )

        print(self.replay_buffer)

            #     if (
            #         self.total_step[agent] % 10 == 0
            #         and self.total_step[agent] > self.initial_step
            #     ):
            #         samples = self.replay_buffer[agent].sample(self.batch_size)
            #         states, actions, new_states, rewards = (
            #             list(),
            #             list(),
            #             list(),
            #             list(),
            #         )
            #         for transition in samples:
            #             states.append(transition[0])
            #             actions.append(transition[1])
            #             new_states.append(transition[2])
            #             rewards.append(transition[3])
            #
            #         targets = rewards + self.gamma * self.q_target[agent].compute_max_q(
            #             new_states
            #         )
            #         loss_in_episode[agent].append(
            #             self.q_principal[agent].train(states, actions, targets)
            #         )
            #
            #         if self.total_step[agent] % self.tau == 0:
            #             run_target_update(self.q_principal[agent], self.q_target[agent])
            #
            #     self.total_step[agent] += 1
            #     reward_sum[agent] += reward
            #
            # reward_record[episode] = reward_sum
            # loss_mean_record[episode] = {
            #     agent: mean(loss) if len(loss) > 0 else None
            #     for agent, loss in loss_in_episode.items()
            # }
            # print(f"Training episode {episode} with reward {reward_sum}.")

        # with open("reward.json", "w") as out_file:
        #     json.dump(reward_record, out_file)
        #
        # with open("loss.json", "w") as out_file:
        #     json.dump(loss_mean_record, out_file)


madqn = MADQN(env=env)
madqn.train()
