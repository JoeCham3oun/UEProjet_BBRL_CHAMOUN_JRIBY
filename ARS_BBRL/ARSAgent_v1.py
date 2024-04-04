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
        obs = self.get(("obs", t))
        action = torch.mm(self.M + self.nu * self.delta, obs.unsqueeze(1)).squeeze()

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
