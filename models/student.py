"""
Student Policy Network. Acts on the current observation + latent dynamics vector z

The student ha the same Gaussian structute as the teacher (diagonal covariance, learned_log_std),
but the input is wider: [obs(30) | z(d_z)] instead of just obs(30)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .build_mlp import build_mlp

class StudentPolicy(nn.Module):
    """
    Diagonal Gaussian policy conditioned on observation + latent z
    """

    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 0.5

    def __init__(self, obs_dim, latent_dim, act_dim, hidden_size=(256,256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.act_dim = act_dim

        # Input size is obs_dim + latent_dim
        self.mean_net = build_mlp(obs_dim + latent_dim, act_dim, hidden_size)

        # State independent log standard deviation
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Small initial output
        with torch.no_grad():
            self.mean_net[-1].weight.data.mul(0.01)
            self.mean_net[-1].bias.data.zero_()

    def _make_dist(self, obs_z):
        """
        Build Gaussian distribution from [obs, z] input
        """
        mean = self.mean_net(obs_z)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)
    
    def forward(self, obs, z):
        combined  = torch.cat([obs, z], dim=-1)
        return self._make_dist(combined)
    
    @torch.no_grad()
    def act(self, obs, z):
        """
        Sample an action for rollout collection with no gradient
        """
        dist = self.forward(obs, z)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate(self, obs, z, actions):
        """
        Evaluate the log-probability and entropy for a batch.
        """
        dist = self.forward(obs, z)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy
    
    def get_mean(self, obs, z):
        """
        Calculates the deterministic mean action. Used as a distillation output
        """
        combined = torch.cat([obs, z], dim=-1)
        return self.mean_net(combined)