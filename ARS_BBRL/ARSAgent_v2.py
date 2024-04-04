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
        obs = self.get(("obs", t))
        M_plus_delta = self.M + self.nu * self.delta
        sigma_sqrt_inv = torch.linalg.inv(torch.sqrt(self.sigma))
        product = torch.mm(M_plus_delta, sigma_sqrt_inv)
        diff = obs - self.mu
        action = torch.mm(product, diff.unsqueeze(1)).squeeze()

        if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
            action_list = [tensor.item() for tensor in action]
            try:
                action = 1 if action_list > 0 else 0
            except ValueError as ve:
                action = np.argmax(action_list)
            except TypeError as te:
              normalized_actions = np.clip(action_list, -1, 1)
              action = 1 if normalized_actions[0] > 0 else 0

        else:
            try:
                action = np.clip(action, self.gym_env.action_space.low, self.gym_env.action_space.high)
            except ValueError as ve:
                action = np.clip(np.argmax(action), self.gym_env.action_space.low, self.gym_env.action_space.high)

        action = torch.tensor(action, dtype=torch.float32)
        action = action.view(-1)
        action = action.to('cpu')
        action = action.float()
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

        # Version-specific update
        states_tensor = torch.tensor(states_encountered)
        self.mu = torch.mean(states_tensor, dim=0)
        states_tensor_transpose = torch.transpose(states_tensor, 0, 1)
        sigma = torch.matmul(states_tensor_transpose, states_tensor) / states_tensor.size(0)
        self.sigma += self.epsilon * torch.eye(sigma.size(0))
