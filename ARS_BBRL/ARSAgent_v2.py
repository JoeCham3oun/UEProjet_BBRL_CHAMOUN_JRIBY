import torch
import numpy as np
import gymnasium as gym

from ARSAgent import ARSAgent

################################################################################
#                                 ARS AGENT V2                                 #
################################################################################
class ARSAgent_v2(ARSAgent):
    def forward(self, t, **kwargs):
        """
        Forward pass of the ARS agent.

        Args:
            t (int): Time step.
            **kwargs: Additional arguments.
        """
        obs = self.get(("env/env_obs", t)).t()
        M_plus_delta = self.M + self.nu * self.delta
        sigma_sqrt_inv = torch.linalg.inv(torch.sqrt(self.sigma))
        product = torch.mm(M_plus_delta, sigma_sqrt_inv)
        diff = obs - self.mu
        action = torch.mm(product, diff)
        if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
            max_values, _ = torch.max(action, dim=1)
            action = torch.argmax(max_values).unsqueeze(0)
        else:
            action = action[:, :1]
        self.set(("action", t), action)


    def update_policy(self, deltas, states_encountered, rewards_plus, rewards_minus):
        """
        Update the policy weights for ARS version 2.

        Args:
            deltas (list): List of perturbation vectors.
            states_encountered (list): List of encountered states.
            rewards_plus (list): Rewards obtained with positive perturbations.
            rewards_minus (list): Rewards obtained with negative perturbations.
        """
        super().update_policy(deltas, states_encountered, rewards_plus, rewards_minus)
        states_encountered = torch.tensor(states_encountered)
        
        # Version-specific update
        self.mu = torch.mean(states_encountered)
        sigma = torch.matmul(states_encountered.T, states_encountered) / states_encountered.size(0)
        self.sigma += self.epsilon * torch.eye(sigma.size(0))
