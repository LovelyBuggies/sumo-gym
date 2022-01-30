import random
import numpy as np

import torch


class ReplayBuffer(object):
    def __init__(self):
        self.memory = list()

    def push(self, trajectory):
        self.memory.append(trajectory)

    def sample(self):
        return random.sample(self.memory, 1)


class QNetwork(object):
    def __init__(self, obs_size, act_size, lr):
        self.obs_size = obs_size
        self.act_size = act_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 28),
            torch.nn.ReLU(),
            torch.nn.Linear(28, 14),
            torch.nn.ReLU(),
            torch.nn.Linear(14, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, act_size)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def compute_Qvalues(self, states, actions):
        states = torch.FloatTensor(states)
        q_preds = self.model(states)
        action_onehot = self._to_one_hot(actions, self.act_size)
        q_preds_selected = torch.sum(q_preds * action_onehot, axis=-1)
        return q_preds_selected

    def compute_maxQvalues(self, states):
        states = torch.FloatTensor(states)
        Qvalues = self.model(states).cpu().data.numpy()
        q_pred_greedy = np.max(Qvalues, 1)
        return q_pred_greedy

    def compute_argmaxQ(self, state):
        state = torch.FloatTensor(state)
        Qvalue = self.model(state).cpu().data.numpy()
        greedy_action = np.argmax(Qvalue.flatten())
        return greedy_action

    def train(self, states, actions, targets):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)
        q_pred_selected = self.compute_Qvalues(states, actions)
        loss = torch.mean((q_pred_selected - targets) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.numpy()

def run_target_update(Q_principal, Q_target):
    for v, v_ in zip(Q_principal.model.parameters(), Q_target.model.parameters()):
        v_.data.copy_(v.data)
