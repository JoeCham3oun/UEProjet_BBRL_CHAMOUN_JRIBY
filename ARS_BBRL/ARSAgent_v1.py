import torch
import numpy as np
import gymnasium as gym

from ARSAgent import ARSAgent

################################################################################
#                                 ARS AGENT V1                                 #
################################################################################
class ARSAgent_v1(ARSAgent):
    def forward(self, t, **kwargs):
        """
        Forward pass of the ARS agent.

        Args:
            t (int): Time step.
            **kwargs: Additional arguments.
        """
        obs = self.get(("env/env_obs", t)).t()
        # thetas = obs.squeeze().tolist()
        # theta = np.arctan2(thetas[1], thetas[0])
        # plot_pendulum(theta)
        action = torch.mm(self.M + self.nu * self.delta, obs)
        if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
            action = action.argmax().unsqueeze(0)
        self.set(("action", t), action)
