import random
import numpy as np

import torch
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_len=10_000):
        self.max_len = max_len
        self.memory = deque()

    def push(self, transition):
        if len(self.memory) >= self.max_len:
            self.memory.popleft()

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(set(self.memory), batch_size)

    def __repr__(self):
        return str(self.memory)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]


class QNetwork(object):
    def __init__(self, observation_size, n_action, lr):
        self.observation_size = observation_size
        self.n_action = n_action
        self.lr = lr
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_size, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 26),
            torch.nn.ReLU(),
            torch.nn.Linear(26, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, self.n_action),
            torch.nn.ReLU(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def compute_q(self, states, actions):
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        action_onehot = self.to_one_hot(actions, self.n_action)
        q_preds_selected = torch.sum(q_preds * action_onehot, axis=-1)
        return q_preds_selected

    def compute_max_q(self, states):
        states = torch.FloatTensor([states])
        q_values = self.model(states).cpu().data.numpy()
        q_pred_greedy = np.max(q_values, 1)
        return q_pred_greedy

    def compute_argmax_q(self, state):
        state = torch.FloatTensor([state])
        qvalue = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(qvalue.flatten())
        return greedy_action

    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_pred_selected = self.compute_q(states, actions)
        loss = torch.mean((q_pred_selected - targets) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.item()


def run_target_update(q_principal, q_target):
    for v, v_ in zip(q_principal.model.parameters(), q_target.model.parameters()):
        v_.data.copy_(v.data)
