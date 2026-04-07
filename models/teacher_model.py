"""
Teacher Actor/Critic Model

Actor Model -> obs_actor (30-dim) -> action_distribution
Critic Model -> obs_actor + privileged_bs -> 37-dim -> Scalar Value
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .build_mlp import build_mlp

class TeacherPolicy(nn.Module):
    """
    Diagonal Gaussian Policy with state independent log-std 
    """

    # HYPERPARAMETER
    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 0.5

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mean_net = build_mlp(obs_dim, act_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Small Initial output -> Actions start near 0
        with torch.no_grad():
            self.mean_net[-1].weight.data.mul_(0.01)
            self.mean_net[-1].bias.zero_()

    def forward(self, obs):
        # Return the normal distribution over actions
        mean = self.mean_net(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)
    
    @torch.no_grad()
    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate(self, obs, actions):
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy
    
    def get_mean(self, obs):
        return self.mean_net(obs)
    

class TeacherValueFn(nn.Module):
    """
    Critic function which outputs the value.
    """
    def __init__(self, obs_dim, privileged_obs_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.net = build_mlp(obs_dim+privileged_obs_dim, 1, hidden_sizes)

    def forward(self, obs_actor, priviliged_obs):
        x = torch.cat([obs_actor, priviliged_obs], dim=-1)
        return self.net(x).squeeze(-1)