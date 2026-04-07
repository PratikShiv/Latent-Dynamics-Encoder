"""
Dynamics randomizer parameters
"""

from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class DynamicsConfig:
    friction_range: Tuple[float, float] = (0.3, 1.5)
    mass_scale_range: Tuple[float, float] = (0.7, 1.4)
    action_delay_range: Tuple[int, int] = (0, 6)
    obs_delay_range: Tuple[int, int] = (0, 10)
    external_force_range: Tuple[float, float] = (0.0, 3.0)

    # how often to re-sample external disturbance force
    force_resample_interval: int = 50

    @property
    def privileged_dim(self) -> int:
        """ Dimension of privileged observation vector:
            - Friction
            - Mass_scale
            - Action Delay
            - Observation Delay
            - force_x
            - foce_y
            - force_z
        """
        return 7
    
    def sample(self, rng):
        # Sample a full set of dynamic parameters
        friction_scale = float(rng.uniform(*self.friction_range))
        mass_scale = float(rng.uniform(*self.mass_scale_range))
        action_delay = int(rng.integers(self.action_delay_range[0], self.action_delay_range[1] + 1))
        obs_delay = int(rng.integers(self.obs_delay_range[0], self.obs_delay_range[1] + 1))
        
        return {
            "friction_scale": friction_scale,
            "mass_scale": mass_scale,
            "action_delay": action_delay,
            "obs_delay": obs_delay
        }