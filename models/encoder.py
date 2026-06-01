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
    def __init__(self, obs_dim, act_dim, history_length, latent_dim,
                 hidden_size=(256,), channels=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.history_length = history_length
        self.latent_dim = latent_dim

        # In each timestep, we have (obs_dim + act_dim) features.
        # H timesteps are flattened into 1 long vector
        self.pair_dim = obs_dim + act_dim

        """
        RMA style 3 layer 1D conv backbone over the time axis
        With H=50, pair_dim=38, the shape progresion is:
            input           : (B, 38, 50)
            Conv1(k=8, s=4) : (B, 64, 11)
            Conv2(k=5, s=1) : (B, 64, 7)
            Conv3(k=5, s=1) : (B, 64, 3)    -> Flatten -> 96 features
        """
        self.conv = nn.Sequential(
            nn.Conv1d(self.pair_dim, channels, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
        )

        # Discover the flattened conv output size by tunning a dummy forward,
        # so the kernel/stride changes don't require manual recompuatation.
        with torch.no_grad():
            dummy = torch.zeros(1, self.pair_dim, self.history_length)
            conv_out_dim = self.conv(dummy).reshape(1, -1).size(1)

        fc_hidden = hidden_size[0] if isinstance(hidden_size, (tuple, list)) \
                                    else int(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, latent_dim),
        )
        
        # Small initial output so z starts near 0
        with torch.no_grad():
            self.head[-1].bias.data.zero_()

    def forward(self, history_flat):
        """
        Histroy_flat -> (batch, history_length*(obs_dim + act_dim))

        Returns z -> (batch, latent_dim). No output activation
        """
        B = history_flat.size(0)
        # (B, H*P) -> (B, H, P) -> (B, P, H) for nn.Conv1d's (B, C, L) convention
        x = history_flat.view(B, self.history_length, self.pair_dim).transpose(1, 2)
        x = self.conv(x)
        x = x.reshape(B, -1)
        return self.head(x)