"""
Adaptation Encoder - Maps proprioceoptive history to a latestn dynamics vector z

The encoder takes the last H timesteps of (observation, action) pairs and compresses
them into a compact latent vector z.

z should implicitly encode the dynamics parameters (mass, friction, delays) that the robot
is currently experiencing, learned purely from the pattern of how joints moved and the body responded.

"""

import torch
import torch.nn as nn

from.build_mlp import build_mlp

class AdaptationEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, history_length, latent_dim, hidden_size=(256,256)):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.history_length = history_length
        self.latent_dim = latent_dim

        # In each timestep, we have (obs_dim + act_dim) features.
        # H timesteps are flattened into 1 long vector
        self.pair_dim = obs_dim + act_dim
        input_dim = history_length * self.pair_dim

        self.net = build_mlp(input_dim, latent_dim, hidden_size)
        # Small initial output so z starts near 0
        with torch.no_grad():
            self.net[-1].bias.data.zero_()

    def forward(self, history_flat):
        """
        Histroy_flat -> (batch, history_length*(obs_dim + act_dim))

        Returns z -> (batch, latent_dim). No output activation
        """
        return self.net(history_flat)