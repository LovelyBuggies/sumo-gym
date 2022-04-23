import os
import numpy as np
import json
import gym
import random
import sys

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
from DQN.dqn import (
    QNetwork,
    LowerQNetwork_ChargingStation,
    LowerQNetwork_Demand,
    ReplayBuffer,
    run_target_update,
)
from statistics import mean

prefix = "cosmos/"

env = gym.make(
    "FMP-v0",
    mode="sumo_config",
    verbose=1,
    sumo_config_path="assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="assets/data/cosmos/cosmos.cs.add.xml",
    render_env=True if str(sys.argv[sys.argv.index("--render") + 1]) == "1" else False,
)


class MADQN(object):
    def __init__(
        self,
        env,
        lr=0.003,
        batch_size=4,
        tau=50,
        episodes=200,
        gamma=0.95,
        epsilon=1.0,
        decay_period=25,
        decay_rate=0.95,
        min_epsilon=0.01,
        initial_step=400,
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
        self.initial_step_lower = 10

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
        if os.path.exists(prefix + "upper-loss.json"):
            os.remove(prefix + "upper-loss.json")

        if os.path.exists(prefix + "upper-reward.json"):
            os.remove(prefix + "upper-reward.json")

        if os.path.exists(prefix + "lower-loss.json"):
            os.remove(prefix + "lower-loss.json")

        if os.path.exists(prefix + "lower-reward.json"):
            os.remove(prefix + "lower-reward.json")

        if os.path.exists(prefix + "metrics.json"):
            os.remove(prefix + "metrics.json")

        with open(prefix + "upper-loss.json", "w") as out_file:
            out_file.write("{")

        with open(prefix + "upper-reward.json", "w") as out_file:
            out_file.write("{")

        with open(prefix + "lower-loss.json", "w") as out_file:
            out_file.write("{")

        with open(prefix + "lower-reward.json", "w") as out_file:
            out_file.write("{")

        with open(prefix + "metrics.json", "w") as out_file:
            out_file.write("{")

    def _wrap_up_output_file(self):
        with open(prefix + "upper-loss.json", "a") as out_file:
            out_file.write("}")

        with open(prefix + "upper-reward.json", "a") as out_file:
            out_file.write("}")

        with open(prefix + "lower-loss.json", "a") as out_file:
            out_file.write("}")

        with open(prefix + "lower-reward.json", "a") as out_file:
            out_file.write("}")

        with open(prefix + "metrics.json", "a") as out_file:
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
                run_target_update(
                    self.q_principal_lower_demand, self.q_target_lower_demand
                )

    def _update_lower_network_cs(self, loss_in_episode_cs):
        if (
            self.lower_total_step_cs % 5 == 0
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
                run_target_update(
                    self.q_principal_upper[agent], self.q_target_upper[agent]
                )

    def _generate_upper_level_action(
        self, agent, upper_last, prev_action_upper, episode_step
    ):
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

    def _calculate_lower_reward(
        self, location, prev_vector, upper_action, lower_action
    ):
        scalar = 50  # to make the number bigger

        if upper_action == 1:  # charge
            dist_to_all_cs = get_dist_to_charging_stations(
                env.fmp.vertices, env.fmp.edges, env.fmp.charging_stations, location
            )
            weight = (
                2
                if len(env.fmp.charging_stations[lower_action].charging_vehicle) + 1
                <= env.fmp.charging_stations[lower_action].n_slot
                else 1
            )
            return (
                scalar
                * weight
                * mean(dist_to_all_cs)
                / (dist_to_all_cs[lower_action] + 1)
            )  # avoid zero division

        elif upper_action == 0:  # demand
            travel_dist = get_dist_of_demands(
                env.fmp.vertices, env.fmp.edges, env.fmp.demands
            )[lower_action]
            total_dist = get_dist_to_finish_demands(
                env.fmp.vertices, env.fmp.edges, env.fmp.demands, location
            )[lower_action]
            return (
                scalar
                * (1 - prev_vector[lower_action])
                * travel_dist
                / (total_dist + 1)
            )  # avoid zero division

    def _generate_lower_level_action(
        self, agent, lower_last, upper_action, episode_step
    ):
        observation, _, done, info = lower_last

        agent_idx = env.agent_name_idx_mapping[agent]

        index = None
        new_state = None
        prev_state = None
        dest_area = None
        actual_reward = 0

        flag = (
            0 if np.random.rand(1) < self.epsilon else 1
        )  # random sample if 0, use network if 1

        if upper_action == 1:  # action to charge
            self.lower_total_step_cs += 1
            index = (
                self.q_principal_lower_cs.compute_argmax_q(observation[1])
                if flag
                else env.action_space_lower_cs().sample()
            )
            prev_state = observation[1]
            new_state = observation[1][:-1]  # list of current ocupancy of cs
            new_state[index] = (
                0
                if observation[1][index] < env.fmp.charging_stations[index].n_slot
                else 1
            )
            dest_area = env.fmp.vertices[env.fmp.charging_stations[index].location].area
            new_state.append(dest_area)
            actual_reward = self._calculate_lower_reward(
                env.fmp.electric_vehicles[agent_idx].location,
                prev_state,
                upper_action,
                index,
            )
            self.replay_buffer_lower_cs.push(
                [tuple(prev_state), index, tuple(new_state), actual_reward]
            )

        elif upper_action == 0:  # action to load
            self.lower_total_step_demand += 1
            index = (
                self.q_principal_lower_demand.compute_argmax_q(observation[0])
                if flag
                else env.action_space_lower_demand().sample()
            )
            prev_state = observation[0]
            new_state = observation[0][:-1]  # demand vector
            new_state[index] = 1
            dest_area = env.fmp.vertices[env.fmp.demands[index].destination].area
            new_state.append(dest_area)
            actual_reward = self._calculate_lower_reward(
                env.fmp.electric_vehicles[agent_idx].location,
                prev_state,
                upper_action,
                index,
            )

            self.replay_buffer_lower_demand.push(
                [tuple(prev_state), index, tuple(new_state), actual_reward]
            )

        return actual_reward, done, index

    def train(self):

        self._initialize_output_file()
        (
            first_line_loss,
            first_line_reward,
            first_line_loss_lower,
            first_line_reward_lower,
            first_metrics,
        ) = (True, True, True, True, True)

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
                (
                    upper_reward,
                    upper_done,
                    upper_action,
                    info,
                ) = self._generate_upper_level_action(
                    agent, upper_last, prev_action_upper, episode_step
                )
                (
                    lower_reward,
                    lower_done,
                    lower_action,
                ) = self._generate_lower_level_action(
                    agent, lower_last, upper_action, episode_step
                )

                final_info = info
                prev_action_upper[agent] = upper_action
                prev_action_lower[agent] = lower_action
                env.step((upper_action, lower_action))
                episode_step[agent] += 1

                self._update_upper_network(agent, upper_done, loss_in_episode_upper)
                self._update_lower_network(
                    loss_in_episode_lower_demand, loss_in_episode_lower_cs
                )

                self.total_step[agent] += 1
                reward_sum_upper[agent] += upper_reward
                reward_sum_lower += lower_reward

            reward_sum_upper_mean = {
                agent: reward_sum_upper[agent] / episode_step[agent]
                for agent in env.possible_agents
            }
            reward_record = {episode: reward_sum_upper_mean}
            reward_record_lower = {
                episode: reward_sum_lower / max(episode_step.values())
            }
            loss_mean_record = {
                episode: {
                    agent: mean(loss) if len(loss) > 0 else None
                    for agent, loss in loss_in_episode_upper.items()
                }
            }
            cs_mean = (
                mean(loss_in_episode_lower_cs) if len(loss_in_episode_lower_cs) else 0
            )
            demand_mean = (
                mean(loss_in_episode_lower_demand)
                if len(loss_in_episode_lower_demand)
                else 0
            )
            loss_mean_reword_lower = {
                episode: {
                    "total_mean": cs_mean / self.lower_total_step_cs
                    + demand_mean / self.lower_total_step_demand,
                    "demand_mean": demand_mean / self.lower_total_step_demand,
                    "cs_mean": cs_mean / self.lower_total_step_cs,
                }
            }
            metric = {episode: final_info.__dict__}

            with open(prefix + "upper-reward.json", "a") as out_file:
                if first_line_reward:
                    first_line_reward = False
                else:
                    out_file.write(",")

                data = json.dumps(reward_record)
                out_file.write(data[1:-1])

            with open(prefix + "upper-loss.json", "a") as out_file:
                if first_line_loss:
                    first_line_loss = False
                else:
                    out_file.write(",")

                data = json.dumps(loss_mean_record)
                out_file.write(data[1:-1])

            with open(prefix + "lower-reward.json", "a") as out_file:
                if first_line_reward_lower:
                    first_line_reward_lower = False
                else:
                    out_file.write(",")

                data = json.dumps(reward_record_lower)
                out_file.write(data[1:-1])

            with open(prefix + "lower-loss.json", "a") as out_file:
                if first_line_loss_lower:
                    first_line_loss_lower = False
                else:
                    out_file.write(",")

                data = json.dumps(loss_mean_reword_lower)
                out_file.write(data[1:-1])

            with open(prefix + "metrics.json", "a") as out_file:
                if first_metrics:
                    first_metrics = False
                else:
                    out_file.write(",")

                data = json.dumps(metric)
                out_file.write(data[1:-1])

            (f"Training episode {episode} with reward {reward_sum_upper}.")

        self._wrap_up_output_file()


madqn = MADQN(env=env)
madqn.train()
