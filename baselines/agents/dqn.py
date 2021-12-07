from .models.noisy_linear import NoisyLinear
import numpy as np
import sys

try:
    import torch
    from torch import nn, optim
except ModuleNotFoundError:
    print(
        "sumo_gym requires torch to train the agent, either install sumo_gym[torch] or torch",
        file=sys.stderr,
    )
    raise


class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, act_dim, n_hid=128, noisy=True, clip_grad_val=None):
        super(DuelingNetwork, self).__init__()

        if isinstance(state_dim, (list, tuple)):
            state_dim = np.prod(state_dim)
            self.flatten = nn.Flatten()
        else:
            self.flatten = nn.Identity()

        self.input_layer = nn.Linear(state_dim, n_hid)
        self.hidden_layer_1 = nn.Linear(n_hid, n_hid)
        self.hidden_layer_2 = nn.Linear(n_hid, n_hid)
        self.hidden_layer_3 = (
            NoisyLinear(n_hid, n_hid) if noisy else nn.Linear(n_hid, 1)
        )

        self.state_value_layer = NoisyLinear(n_hid, 1) if noisy else nn.Linear(n_hid, 1)
        self.action_value_layer = (
            NoisyLinear(n_hid, act_dim) if noisy else nn.Linear(n_hid, act_dim)
        )

        self.noisy_layers = [
            "state_value_layer",
            "action_value_layer",
            "hidden_layer_3",
        ]

        self.ReLU = nn.ReLU()
        self.clip_grad_val = clip_grad_val

    def forward(self, states):
        states = self.flatten(states)

        x = self.ReLU(self.input_layer(states))
        x = self.ReLU(self.hidden_layer_1(x))
        x = self.ReLU(self.hidden_layer_2(x))

        state_value = self.state_value_layer(x)
        action_value = self.action_value_layer(x)

        # Q(s, a) = V(s) + A(s, a) - sum_{a \in A}{A(s, a)} / |A|
        return (
            state_value + action_value - torch.mean(action_value, dim=-1, keepdims=True)
        )

    def train_step(self, loss, optim, lr_scheduler=None):
        optim.zero_grad()
        loss.backward()

        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        optim.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss

    def reset_noise(self):
        for name, module in self.named_children():
            if name in self.noisy_layers:
                module.reset_noise()

    def polyak_update(self, source_network, source_ratio=0.5):
        """
        Update the parameters of the network with the parameters of the source network.
        source_ratio = 1.0 simply performs a copy, rather than a polyak update.
        """

        for src_param, param in zip(source_network.parameters(), self.parameters()):
            param.data.copy_(
                source_ratio * src_param.data + (1.0 - source_ratio) * param.data
            )


class DQN(object):
    def __init__(
        self,
        state_dim,
        act_dim,
        gamma,
        n_hid=64,
        lr=1e-4,
        epsilon=0.9,
        tau=1,
        device=None,
        clip_grad_val=None,
    ):
        """
        Baseline implementatioin of Q-Function:

        - Collect sample transitions from the environment (store in a replay memory)
        - sample batches of experience (from replay memory)
        - For each transition, calculate the 'target' value (current estimate of state-action value)
                y_t = r_t + gamma * max_a Q(s_{t+1}, a)
        - Calculate the estimate of the state-action values.
                y_hat_t = Q(s_t, a_t)
        - Calculate the loss, L(y_hat_t, y_t) - m.s.e (td-error)
        - Compute the gradient of L with respect to the network parameters, and take a gradient descent step.
        """

        self.act_dim = act_dim
        self.state_dim = state_dim
        self.gamma = gamma

        self.epsilon = epsilon
        self.tau = tau
        self.epsilon_decay_rate = 0.9995
        self.tau_decay_rate = 0.9995
        self.frozen_epsilon = epsilon
        self.frozen_tau = tau
        self.min_epsilon = 0.05
        self.min_tau = 0.1

        self.n_hid = n_hid
        self.lr = lr
        self.huber_loss = nn.HuberLoss(reduction="none", delta=1.0)
        self.mse_loss = nn.MSELoss(reduction="none")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        self.clip_grad_val = clip_grad_val
        self.init_network()

    def init_network(self, network=None):
        self.model = DuelingNetwork(
            self.state_dim, self.act_dim, self.n_hid, clip_grad_val=self.clip_grad_val
        ).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def get_loss(self, experience, IS_weights=None):
        states, actions = experience["states"], experience["actions"]
        q_preds = self.model(states).gather(dim=-1, index=actions.long()).squeeze(-1)
        q_targets = self.compute_targets(experience)

        q_loss = self.mse_loss(q_preds, q_targets)
        errors = (q_preds.detach() - q_targets).abs().cpu().numpy()

        if IS_weights is not None:
            q_loss = torch.multiply(IS_weights, q_loss)
            q_loss = torch.mean(q_loss)
        else:
            q_loss = torch.mean(q_loss)

        return q_loss, errors

    def compute_targets(self, experience):
        with torch.no_grad():
            q_preds_next = self.model(experience["next_states"])

        max_q_preds, _ = q_preds_next.max(dim=-1, keepdim=False)
        q_targets = experience["rewards"] + self.gamma * max_q_preds * (
            1 - experience["dones"]
        )
        return q_targets

    def train_network(self, experience, IS_weights=None):
        # Compute the Q-targets and TD-Error

        avg_loss = 0.0
        loss, error = self.get_loss(experience, IS_weights)
        self.model.train_step(loss, self.optim, self.lr_scheduler)

        return loss.item(), error

    def act_epsilon_greedy(self, state):
        # With probability epsilon, take a random action
        action_values = self.model(state).detach()

        if np.random.rand() < self.epsilon:
            logits = torch.ones_like(action_values).to(self.device)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            return action.cpu().squeeze().numpy()

        action = action_values.argmax()
        return action.cpu().squeeze().numpy()

    def act_boltzmann(self, state):
        action_values = self.model(state).detach()

        # sample from a Boltzman distribution over the state-action values
        logits = self.model(state) / self.tau
        action_pd = torch.distributions.Categorical(logits=logits)
        action = action_pd.sample()
        return action.squeeze().numpy()

    def act_greedy(self, state):
        action_values = self.model(state).detach()
        action = action_values.argmax()
        return action.cpu().squeeze().numpy()

    def act(self, state, policy="boltzmann"):
        if not torch.is_tensor(state):
            state = torch.Tensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if policy == "boltzmann":
                action = self.act_boltzmann(state)
            elif policy == "epsilon_greedy":
                action = self.act_epsilon_greedy(state)
            else:
                action = self.act_greedy(state)

        return action

    def update(self, count):
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        self.tau = max(self.tau * self.tau_decay_rate, self.min_tau)

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
