# Latent-Dynamics-Encoder

An adaptation encoder that learns a compact latent representation 
of robot dynamics (mass, friction, actuator delay) from proprioception 
alone — no privileged simulation access at deployment.

Reproduces and extends RMA (Kumar et al., 2021) with explicit 
latent space analysis: UMAP visualizations show smooth manifolds 
across the dynamics distribution.