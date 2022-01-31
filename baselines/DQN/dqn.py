import random
import numpy as np

import torch
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_len=10_000):
        self.max_len = max_len
        self.memory = deque()
        self.sample_w = deque()

    def push(self, transition):
        if len(self.memory) >= self.max_len:
            self.memory.popleft()
            self.sample_w.popleft()

        self.memory.append(transition)
        self.sample_w.append(np.power(abs(transition[3]), 1 / 10))

    def sample(self, batch_size):
        sum_w = sum(self.sample_w)
        return np.choice(self.memory, batch_size, p=[w / sum_w for w in self.sample_w])

    def __repr__(self):
        return str(self.memory)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]


class QNetwork(object):
    def __init__(self, observation_size, action_size, lr):
        self.observation_size = observation_size
        self.action_size = action_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(observation_size, 28),
            torch.nn.ReLU(),
            torch.nn.Linear(28, 14),
            torch.nn.ReLU(),
            torch.nn.Linear(14, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, action_size),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def compute_q(self, states, actions):
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        action_onehot = self.to_one_hot(actions, self.action_size)
        q_preds_selected = torch.sum(q_preds * action_onehot, axis=-1)
        return q_preds_selected

    def compute_max_q(self, states):
        states = torch.FloatTensor(states)
        qvalues = self.model(states).cpu().data.numpy()
        q_pred_greedy = np.max(qvalues, 1)
        return q_pred_greedy

    def compute_argmax_q(self, state):
        state = torch.FloatTensor(state)
        qvalue = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(qvalue.flatten())
        return greedy_action

    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_pred_selected = self.compute_qvalues(states, actions)
        loss = torch.mean((q_pred_selected - targets) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.numpy()


def run_target_update(q_principal, q_target):
    for v, v_ in zip(q_principal.model.parameters(), q_target.model.parameters()):
        v_.data.copy_(v.data)
