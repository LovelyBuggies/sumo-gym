import os
import numpy as np
import json
import gym
import random

import sumo_gym
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    ChargingStation,
    ElectricVehicles,
    get_dist_of_demands,
    get_dist_to_charging_stations,
    get_dist_to_finish_demands,
    get_safe_indicator,
)
from DQN.dqn import QNetwork, LowerQNetwork_ChargingStation, LowerQNetwork_Demand, ReplayBuffer, run_target_update
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
n_electric_vehicle = 1
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
        decay_period=25,
        decay_rate=0.95,
        min_epsilon=0.01,
        initial_step=200,
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
        self.initial_step_lower = 3

        self.q_principal_upper = {
            agent: QNetwork(
                1,
                env.action_space(agent).n,
                self.lr,
            )
            for agent in self.env.agents
        }
        self.q_target_upper = {
            agent: QNetwork(
                1,
                env.action_space(agent).n,
                self.lr,
            )
            for agent in self.env.agents
        }

        self.q_principal_lower_demand = LowerQNetwork_Demand(
            env.action_space_lower_demand().n_demand,
            env.action_space_lower_demand().n_demand,
            self.lr,
        )
        self.q_target_lower_demand = LowerQNetwork_Demand(
            env.action_space_lower_demand().n_demand,
            env.action_space_lower_demand().n_demand,
            self.lr,
        )

        self.q_principal_lower_cs = LowerQNetwork_ChargingStation(
            env.action_space_lower_cs().n_cs,
            env.action_space_lower_cs().n_cs,
            self.lr,
        )
        self.q_target_lower_cs = LowerQNetwork_ChargingStation(
            env.action_space_lower_cs().n_cs,
            env.action_space_lower_cs().n_cs,
            self.lr,
        )

        self.replay_buffer_upper = {
            agent: ReplayBuffer() for agent in self.env.possible_agents
        }
        self.replay_buffer_lower_demand = ReplayBuffer()
        self.replay_buffer_lower_cs = ReplayBuffer()

        self.total_step = {agent: 0 for agent in self.env.possible_agents}
        self.lower_total_step_demand = 0
        self.lower_total_step_cs = 0

    def _initialize_output_file(self):
        if os.path.exists("loss.json"):
            os.remove("loss.json")

        if os.path.exists("reward.json"):
            os.remove("reward.json")

        if os.path.exists("loss_lower.json"):
            os.remove("loss_lower.json")

        if os.path.exists("reward_lower.json"):
            os.remove("reward_lower.json")

        if os.path.exists("metrics.json"):
            os.remove("metrics.json")

        with open("reward.json", "w") as out_file:
            out_file.write("{")

        with open("loss.json", "w") as out_file:
            out_file.write("{")

        with open("reward_lower.json", "w") as out_file:
            out_file.write("{")

        with open("loss_lower.json", "w") as out_file:
            out_file.write("{")

        with open("metrics.json", "w") as out_file:
            out_file.write("{")

    def _wrap_up_output_file(self):
        with open("reward.json", "a") as out_file:
            out_file.write("}")

        with open("loss.json", "a") as out_file:
            out_file.write("}")

        with open("reward_lower.json", "a") as out_file:
            out_file.write("}")

        with open("loss_lower.json", "a") as out_file:
            out_file.write("}")

        with open("metrics.json", "a") as out_file:
            out_file.write("}")

    def _update_lower_network_demand(self, loss_in_episode_demand):
        if (
            self.lower_total_step_demand % 3 == 0
            and self.lower_total_step_demand > self.initial_step_lower
        ):
            samples = self.replay_buffer_lower_demand.sample(self.batch_size)
            states, actions, new_states, rewards = (
                list(),
                list(),
                list(),
                list(),
            )
            for transition in samples:
                states.append(list(transition[0]))
                actions.append(transition[1])
                new_states.append(list(transition[2]))
                rewards.append(transition[3])

            targets = rewards + self.gamma * self.q_target_lower_demand.compute_max_q(
                new_states
            )
            loss_in_episode_demand.append(
                self.q_principal_lower_demand.train(states, actions, targets)
            )

            if self.lower_total_step_demand % self.tau == 0:
                run_target_update(self.q_principal_lower_demand, self.q_target_lower_demand)

    def _update_lower_network_cs(self, loss_in_episode_cs):
        if (
            self.lower_total_step_cs % 3 == 0
            and self.lower_total_step_cs > self.initial_step_lower
        ):
            samples = self.replay_buffer_lower_cs.sample(self.batch_size)
            states, actions, new_states, rewards = (
                list(),
                list(),
                list(),
                list(),
            )
            for transition in samples:
                states.append(list(transition[0]))
                actions.append(transition[1])
                new_states.append(list(transition[2]))
                rewards.append(transition[3])

            targets = rewards + self.gamma * self.q_target_lower_cs.compute_max_q(
                new_states
            )
            loss_in_episode_cs.append(
                self.q_principal_lower_cs.train(states, actions, targets)
            )

            if self.lower_total_step_cs % self.tau == 0:
                run_target_update(self.q_principal_lower_cs, self.q_target_lower_cs)

    def _update_lower_network(self, loss_in_episode_demand, loss_in_episode_cs):
        self._update_lower_network_demand(loss_in_episode_demand)
        self._update_lower_network_cs(loss_in_episode_cs)

    def _update_upper_network(self, agent, done, loss_in_episode):

        if done:
            agent_idx = env.agent_name_idx_mapping[agent]
            self.replay_buffer_upper[agent][-1][2] = get_safe_indicator(
                env.fmp.vertices,
                env.fmp.edges,
                env.fmp.demands,
                env.fmp.charging_stations,
                env.fmp.electric_vehicles[agent_idx].location,
                env.fmp.electric_vehicles[agent_idx].battery,
            )

        if (
            self.total_step[agent] % 10 == 0
            and self.total_step[agent] > self.initial_step
        ):
            samples = self.replay_buffer_upper[agent].sample(self.batch_size)
            states, actions, new_states, rewards = (
                list(),
                list(),
                list(),
                list(),
            )
            for transition in samples:
                states.append(transition[0])
                actions.append(transition[1])
                new_states.append(transition[2])
                rewards.append(transition[3])

            targets = rewards + self.gamma * self.q_target_upper[agent].compute_max_q(
                new_states
            )
            loss_in_episode[agent].append(
                self.q_principal_upper[agent].train(states, actions, targets)
            )

            if self.total_step[agent] % self.tau == 0:
                run_target_update(self.q_principal_upper[agent], self.q_target_upper[agent])

    def _generate_upper_level_action(self, agent, upper_last, prev_action_upper, episode_step):
        observation, reward, done, info = upper_last
        if (
            observation != 3
            and prev_action_upper[agent] is not None
            and prev_action_upper[agent] != 2
        ):
            if self.replay_buffer_upper[agent] and episode_step != 0:
                self.replay_buffer_upper[agent][-1][2] = observation

            self.replay_buffer_upper[agent].push(
                [observation, prev_action_upper[agent], None, reward]
            )

        if observation == 3:
            action = 2
        elif np.random.rand(1) < self.epsilon:
            action = env.action_space(agent).sample()
        else:
            action = self.q_principal_upper[agent].compute_argmax_q(observation)

        return reward, done, action, info

    def _calculate_lower_reward(self, location, prev_vector, upper_action, lower_action):
        scalar = 50 # to make the number bigger

        if upper_action == 1: # charge
            dist_to_all_cs = get_dist_to_charging_stations(env.fmp.vertices, env.fmp.edges, env.fmp.charging_stations, location)
            weight = 2 if len(env.fmp.charging_stations[lower_action].charging_vehicle) + 1 <= env.fmp.charging_stations[lower_action].n_slot else 1
            return scalar * weight * mean(dist_to_all_cs) / (dist_to_all_cs[lower_action] + 1) # avoid zero division

        elif upper_action == 0: # demand
            travel_dist = get_dist_of_demands(env.fmp.vertices, env.fmp.edges, env.fmp.demands)[lower_action]
            total_dist = get_dist_to_finish_demands(env.fmp.vertices, env.fmp.edges, env.fmp.demands, location)[lower_action]
            return scalar * (1-prev_vector[lower_action]) * travel_dist / (total_dist + 1) # avoid zero division


    def _generate_lower_level_action(self, agent, lower_last, upper_action, episode_step):
        observation, _, done, info = lower_last

        agent_idx = env.agent_name_idx_mapping[agent]

        index = None
        new_state = None
        prev_state = None
        dest_area = None
        actual_reward = 0

        flag = 0 if np.random.rand(1) < self.epsilon else 1 # random sample if 0, use network if 1

        if upper_action == 1: # action to charge
            self.lower_total_step_cs += 1
            index = self.q_principal_lower_cs.compute_argmax_q(observation[1]) if flag else env.action_space_lower_cs().sample()
            prev_state = observation[1]
            new_state = observation[1][:-1] # list of current ocupancy of cs
            new_state[index] = 0 if observation[1][index] < env.fmp.charging_stations[index].n_slot else 1
            dest_area = env.fmp.vertices[env.fmp.charging_stations[index].location].area
            new_state.append(dest_area)
            actual_reward = self._calculate_lower_reward(env.fmp.electric_vehicles[agent_idx].location, prev_state, upper_action, index)
            self.replay_buffer_lower_cs.push([tuple(prev_state), index, tuple(new_state), actual_reward])

        elif upper_action == 0: # action to load
            self.lower_total_step_demand += 1
            index = self.q_principal_lower_demand.compute_argmax_q(observation[0]) if flag else env.action_space_lower_demand().sample()
            prev_state = observation[0]
            new_state = observation[0][:-1] # demand vector
            new_state[index] = 1
            dest_area = env.fmp.vertices[env.fmp.demands[index].destination].area
            new_state.append(dest_area)
            actual_reward = self._calculate_lower_reward(env.fmp.electric_vehicles[agent_idx].location, prev_state, upper_action, index)

            self.replay_buffer_lower_demand.push([tuple(prev_state), index, tuple(new_state), actual_reward])   

        return actual_reward, done, index

    def _initialize_output_file(self):
        if os.path.exists("loss.json"):
            os.remove("loss.json")

        if os.path.exists("reward.json"):
            os.remove("reward.json")

        with open("reward.json", "w") as out_file:
            out_file.write("{")

        with open("loss.json", "w") as out_file:
            out_file.write("{")

    def _wrap_up_output_file(self):
        with open("reward.json", "a") as out_file:
            out_file.write("}")

        with open("loss.json", "a") as out_file:
            out_file.write("}")

    def train(self):

        self._initialize_output_file()
        first_line_loss, first_line_reward, first_line_loss_lower, first_line_reward_lower, first_metrics = True, True, True, True, True

        for episode in range(self.episodes):
            env.reset()
            episode_step = {agent: 0 for agent in env.possible_agents}
            reward_sum_upper = {agent: 0 for agent in env.possible_agents}
            reward_sum_lower = 0
            loss_in_episode_upper = {agent: list() for agent in env.possible_agents}
            loss_in_episode_lower_demand = list()
            loss_in_episode_lower_cs = list()
            final_info = {}

            if episode % self.decay_period == 0:
                self.epsilon *= self.decay_rate
                self.epsilon = max(self.min_epsilon, self.epsilon)

            prev_action_upper = {agent: None for agent in env.possible_agents}
            prev_action_lower = {agent: None for agent in env.possible_agents}
            for agent in env.agent_iter():
                upper_last, lower_last = env.last()
                upper_reward, upper_done, upper_action, info = self._generate_upper_level_action(agent, upper_last, prev_action_upper, episode_step)
                lower_reward, lower_done, lower_action = self._generate_lower_level_action(agent, lower_last, upper_action, episode_step)

                final_info = info
                prev_action_upper[agent] = upper_action
                prev_action_lower[agent] = lower_action
                env.step((upper_action, lower_action))
                episode_step[agent] += 1

                self._update_upper_network(agent, upper_done, loss_in_episode_upper)
                self._update_lower_network(loss_in_episode_lower_demand, loss_in_episode_lower_cs)

                self.total_step[agent] += 1
                reward_sum_upper[agent] += upper_reward
                reward_sum_lower += lower_reward

            reward_sum_upper_mean = {agent : reward_sum_upper[agent] / self.total_step[agent] for agent in env.possible_agents}
            reward_record = {episode: reward_sum_upper_mean}
            reward_record_lower = {episode: reward_sum_lower / max(self.total_step.values())}
            loss_mean_record = {
                episode: {
                    agent: mean(loss) if len(loss) > 0 else None
                    for agent, loss in loss_in_episode_upper.items()
                }
            }
            cs_mean = mean(loss_in_episode_lower_cs) if len(loss_in_episode_lower_cs) else 0
            demand_mean = mean(loss_in_episode_lower_demand) if len(loss_in_episode_lower_demand) else 0
            loss_mean_reword_lower = {
                episode: {
                    "total_mean": cs_mean / self.lower_total_step_cs + demand_mean / self.lower_total_step_demand,
                    "demand_mean": demand_mean / self.lower_total_step_demand,
                    "cs_mean": cs_mean / self.lower_total_step_cs
                }
            }
            metric = {
                episode: final_info.__dict__
            }

            with open("reward.json", "a") as out_file:
                if first_line_reward:
                    first_line_reward = False
                else:
                    out_file.write(",")

                data = json.dumps(reward_record)
                out_file.write(data[1:-1])

            with open("loss.json", "a") as out_file:
                if first_line_loss:
                    first_line_loss = False
                else:
                    out_file.write(",")

                data = json.dumps(loss_mean_record)
                out_file.write(data[1:-1])

            with open("reward_lower.json", "a") as out_file:
                if first_line_reward_lower:
                    first_line_reward_lower = False
                else:
                    out_file.write(",")

                data = json.dumps(reward_record_lower)
                out_file.write(data[1:-1])

            with open("loss_lower.json", "a") as out_file:
                if first_line_loss_lower:
                    first_line_loss_lower = False
                else:
                    out_file.write(",")

                data = json.dumps(loss_mean_reword_lower)
                out_file.write(data[1:-1])

            with open("metrics.json", "a") as out_file:
                if first_metrics:
                    first_metrics = False
                else:
                    out_file.write(",")

                data = json.dumps(metric)
                out_file.write(data[1:-1])

            print(f"Training episode {episode} with reward {reward_sum_upper}.")

        self._wrap_up_output_file()


madqn = MADQN(env=env)
madqn.train()
