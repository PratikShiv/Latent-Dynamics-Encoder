"""
Build a MLP based on inputs. This is a common script to modularize codebase.
"""

import torch
import torch.nn as nn

def build_mlp(in_dim, out_dim, hidden_size, activation=nn.Tanh):
    """
    Build the Neural Network
    """
    layers = []
    prev = in_dim

    for h in hidden_size:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))

    return nn.Sequential(*layers)