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

from .dqn import NoisyLinear, DQN


class DDQN(DQN):
    '''
        Double Deep Q-Networks:
        Based on the extension to DQN from the paper:
            Deep Reinforcement Learning with Double Q-learning, 2015
            Hado van Hasselt and Arthur Guez and David Silver
            https://arxiv.org/pdf/1509.06461.pdf

        When calculating the training target for our Q-Network, we utilize a second network (the target network) to
        evaluate the actions selected by the 'online' network which is used to select actions.

        The new 'target' value calculation:
                y_t = r_t + gamma * Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))
    '''

    def __init__(self, state_dim, action_dim, gamma, n_hid=64, lr=1e-4, device=None, noisy_networks=True,
                 target_update_freq=100, clip_grad_val=None):
        self.target_update_freq = target_update_freq
        self.noisy_network = noisy_networks

        super().__init__(state_dim, action_dim, gamma, device=device, n_hid=n_hid, lr=lr, clip_grad_val=clip_grad_val)

    def init_network(self):
        '''
        Initialize the neural network related objects used to learn the policy function
        '''

        self.model = DuelingNetwork(self.state_dim, self.act_dim, self.n_hid, noisy=self.noisy_network,
                                    clip_grad_val=self.clip_grad_val).to(self.device)
        self.target = DuelingNetwork(self.state_dim, self.act_dim, self.n_hid, noisy=self.noisy_network,
                                     clip_grad_val=self.clip_grad_val).to(self.device)
        self.target.polyak_update(source_network=self.model, source_ratio=1.0)  # copy parameters from model to target

        # We do not need to compute the gradients of the target network. It will be periodically
        # updated using the parameters in the online network.
        for param in self.target.parameters():
            param.requires_grad = False

        self.model.train()
        self.target.train()

        self.online_network = self.model
        self.eval_network = self.target

        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def compute_targets(self, experience):
        with torch.no_grad():
            online_q_preds = self.online_network(experience["next_states"])  # online network selects actions

            # Resample parameter noise if using noisy networks.
            if self.noisy_network:
                self.eval_network.reset_noise()

            eval_q_preds = self.eval_network(experience["next_states"])  # Target network evaluates actions

        online_actions = online_q_preds.argmax(dim=-1, keepdim=True)
        next_q_preds = eval_q_preds.gather(-1, online_actions).squeeze(-1)
        q_targets = experience['rewards'] + self.gamma * (1 - experience['dones']) * next_q_preds
        return q_targets

    def update_target_network(self, train_step, ratio=0.5):
        if train_step % self.target_update_freq == 0:
            self.target.polyak_update(source_network=self.model, source_ratio=ratio)

    def reset_noise(self):
        self.online_network.reset_noise()

    def update(self, count):
        self.update_target_network(count)
        super().update(count)