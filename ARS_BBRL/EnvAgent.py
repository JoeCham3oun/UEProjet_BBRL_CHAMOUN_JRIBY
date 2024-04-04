#try:
#    from easypip import easyimport, easyinstall, is_notebook
#except ModuleNotFoundError as e:
#    import subprocess
#    subprocess.run(["pip", "install", "easypip>=1.2.0"])
#    from easypip import easyimport, easyinstall, is_notebook
#
#easyinstall("bbrl>=0.2.2")
#easyinstall("gymnasium")
#easyinstall("bbrl_gymnasium>=0.2.0")
#easyinstall("tensorboard")
#
#try:
#    import bbrl
#except ImportError:
#    import subprocess
#    subprocess.run(["pip", "install", "git+https://github.com/osigaud/bbrl.git"])
#    import bbrl

import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from bbrl.agents.agent import Agent

################################################################################
#                                   ENV AGENT                                  #
################################################################################
class EnvAgent(Agent):
    def __init__(self, cfg):
        """
        Initialize the environment agent.

        Args:
            cfg (DictConfig): Configuration for the agent.
        """
        super().__init__()
        self.gym_env = gym.make(cfg.gym_env.env_name)
        self.states_encountered = []


    def forward(self, t, **kwargs):
        """
        Forward pass of the environment agent.

        Args:
            t (int): Time step.
            **kwargs: Additional arguments.
        """
        if t==0:
            obs = self.gym_env.reset()[0]
            self.set(("obs", t), torch.tensor(obs))
        else:
            action = self.get(("action", t-1))
            # print(f"{t=}")
            # print(f"{action=}")
            if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
                obs, reward, terminated, truncated, _ = self.gym_env.step(int(action.item()))
            else:
                obs, reward, terminated, truncated, _ = self.gym_env.step(action)
            # print(f"{obs=}")
            # print(f"{reward=}")
            # print()
            self.set(("obs", t), torch.tensor(obs))
            # theta = np.arctan2(obs[1], obs[0])
            # plot_pendulum(theta)
            if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
                reward = torch.tensor(reward, dtype=torch.float32)
                reward = action.view(-1)
                reward = action.to('cpu')
                reward = action.float()
                self.set(("reward", t), reward)
            else:
                self.set(("reward", t), reward.unsqueeze(0).clone().detach())
        self.states_encountered.append(obs)


    def get_obs_and_actions_sizes(self):
        """Get observation and action sizes of the environment."""
        observation_dim = self.gym_env.observation_space.shape[0]
        if isinstance(self.gym_env.action_space, gym.spaces.Discrete):
            action_dim = self.gym_env.action_space.n
        else:
            action_dim = self.gym_env.action_space.shape[0]
        return observation_dim, action_dim
        
        
################################################################################


def plot_pendulum(theta):
    x = np.sin(theta)
    y = -np.cos(theta)

    plt.figure(figsize=(5, 5))
    plt.plot([0, x], [0, y], color='black')
    plt.scatter([x], [y], s=100, color='blue', zorder=5)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axvline(x=0, color='grey', lw=1, zorder=0)
    plt.axhline(y=0, color='grey', lw=1, zorder=0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.title('Pendulum State')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
