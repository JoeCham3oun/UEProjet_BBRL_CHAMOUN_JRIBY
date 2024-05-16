import torch
import numpy as np
import gymnasium as gym

from ARSAgent import ARSAgent

################################################################################
#                                 ARS AGENT V1                                 #
################################################################################
class ARSAgent_v1(ARSAgent):
    """
    ARSAgent_v1 class representing an earlier version of the Augmented Random Search (ARS) agent.

    Inherits from:
        ARSAgent (ARSAgent.ARSAgent): Base class for ARS agents.

    Attributes:
        Inherits attributes from ARSAgent base class.
    """
    
    def forward(self, t, **kwargs):
        """
        Forward pass of the ARS agent for ARS version 1.

        Args:
            t (int): Time step.
            **kwargs: Additional arguments.
        """
        obs = self.get(("env/env_obs", t)).t()
        action = torch.mm(self.M + self.nu * self.delta, obs)
        if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
            action = action.argmax().unsqueeze(0)
        self.set(("action", t), action)
