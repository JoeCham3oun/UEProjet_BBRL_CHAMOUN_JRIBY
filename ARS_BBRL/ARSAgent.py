import torch
import numpy as np

from bbrl.agents.agent import Agent

################################################################################
#                                   ARS AGENT                                  #
################################################################################
class ARSAgent(Agent):
    def __init__(self, gym_env, cfg, M):
        """
        Initialize the ARS (Augmented Random Search) agent.

        Args:
            cfg (DictConfig): Configuration for the agent.
            M (torch.tensor): Policy weights.
        """
        super().__init__()
        self.gym_env = gym_env
        self.N = cfg.algorithm.N
        self.M = torch.tensor(M, dtype=torch.float32)
        self.nu = cfg.algorithm.nu
        self.b = cfg.algorithm.b
        self.alpha = cfg.algorithm.alpha
        self.epsilon = cfg.algorithm.epsilon
        self.sigma = torch.eye(M.shape[1])
        self.mu = torch.zeros(M.shape[1])
        self.delta = torch.zeros_like(self.M)


    def forward(self, t, **kwargs):
        """
        Forward pass of the ARS agent.

        Args:
            t (int): Time step.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError("forward must be implemented in subclasses.")


    def set_delta(self, delta):
        """
        Set the perturbation vector.

        Args:
            delta (torch.tensor): Perturbation vector.
        """
        self.delta = torch.tensor(delta, dtype=torch.float32)


    def reset_delta(self):
        """Reset the perturbation vector."""
        self.delta = torch.zeros_like(self.M)


    def update_policy(self, deltas, states_encountered, rewards_plus, rewards_minus):
        """
        Update the policy weights.

        Args:
            deltas (list): List of perturbation vectors.
            states_encountered (list): List of encountered states.
            rewards_plus (list): Rewards obtained with positive perturbations.
            rewards_minus (list): Rewards obtained with negative perturbations.
        """
        scores = list(zip(deltas, rewards_plus, rewards_minus))
        scores.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        
        top_scores = scores[:self.b]

        # Update policy weights using the top b perturbations
        update_step = np.zeros_like(self.M.numpy())
        sigma_rewards = np.std([r for _, r_plus, r_minus in top_scores for r in (r_plus, r_minus)]) + self.epsilon
        
        for delta, reward_plus, reward_minus in top_scores:
            update_step += (reward_plus - reward_minus) * delta

        # Apply update to policy weights
        self.M += (self.alpha / (self.b * sigma_rewards)) * torch.from_numpy(update_step)
