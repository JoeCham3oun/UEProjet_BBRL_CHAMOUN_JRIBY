try:
    from easypip import easyimport, easyinstall, is_notebook
except ModuleNotFoundError as e:
    import subprocess
    subprocess.run(["pip", "install", "easypip>=1.2.0"])
    from easypip import easyimport, easyinstall, is_notebook

easyinstall("bbrl>=0.2.2")
easyinstall("gymnasium")
easyinstall("bbrl_gymnasium>=0.2.0")
easyinstall("tensorboard")

try:
    import bbrl
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "git+https://github.com/osigaud/bbrl.git"])
    import bbrl

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent

from ARSAgent_v1 import ARSAgent_v1
from ARSAgent_v2 import ARSAgent_v2
from EnvAgent import EnvAgent

# Create the ARS Agent
def create_ARS_agent(cfg, env_agent):
    observation_dim, action_dim = env_agent.get_obs_and_actions_sizes()
    M = np.zeros((action_dim, observation_dim))

    if cfg.algorithm.v2:
        ars_agent = ARSAgent_v2(env_agent.gym_env, cfg, M=M)
    else:
        ars_agent = ARSAgent_v1(env_agent.gym_env, cfg, M=M)

    return ars_agent


def init_env_and_agent(cfg):
    env_agent = EnvAgent(cfg)
    ars_agent = create_ARS_agent(cfg, env_agent)
    composed_agent = Agents(env_agent, ars_agent)
    t_agent = TemporalAgent(composed_agent)
    workspace = Workspace()

    return env_agent, ars_agent, composed_agent, t_agent, workspace


def run_episode_with_perturbation(ars_agent, delta, workspace, t_agent, H):
    ars_agent.set_delta(delta)
    
    workspace.clear()
    t_agent(workspace, t=0, n_steps=H)
    reward = workspace['reward'].sum()
    ars_agent.reset_delta()

    return reward


def display_results(rewards_plus_logs, rewards_minus_logs, best_rewards_log, std_rewards_log, num_episodes):
    plt.figure(figsize=(10, 6))
    episodes = range(num_episodes)
    
    plt.plot(rewards_plus_logs, label='Average Reward Plus')
    plt.plot(rewards_minus_logs, label='Average Reward Minus')
    plt.plot(best_rewards_log, label='Best Reward', color='green')
    plt.fill_between(episodes, np.array(rewards_plus_logs) - np.array(std_rewards_log), np.array(rewards_plus_logs) + np.array(std_rewards_log), color='blue', alpha=0.1)
    plt.fill_between(episodes, np.array(rewards_minus_logs) - np.array(std_rewards_log), np.array(rewards_minus_logs) + np.array(std_rewards_log), color='orange', alpha=0.1)
    plt.title('ARS Performance over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_ars(cfg: DictConfig):
    env_agent, ars_agent, composed_agent, t_agent, workspace = init_env_and_agent(cfg)
    
    H = env_agent.gym_env.spec.max_episode_steps

    best_reward = -np.inf
    rewards_plus_logs = []
    rewards_minus_logs = []
    best_rewards_log = []
    std_rewards_log = []
    episode_rewards = []

    for episode in range(cfg.algorithm.num_episodes):
        deltas = np.random.randn(ars_agent.N, *ars_agent.M.shape)
        rewards_plus = []     # To store rewards when adding perturbations
        rewards_minus = []    # To store rewards when subtracting perturbations
        states_encountered = []

        for delta in deltas:
            rewards = run_episode_with_perturbation(ars_agent, delta, workspace, t_agent, H)
            rewards_plus.append(rewards)
            states_encountered.extend(env_agent.states_encountered)

            rewards = run_episode_with_perturbation(ars_agent, -delta, workspace, t_agent, H)
            rewards_minus.append(rewards)
            states_encountered.extend(env_agent.states_encountered)
            
        # Calculate statistics for logging
        average_reward_plus = np.mean(rewards_plus)
        average_reward_minus = np.mean(rewards_minus)
        std_reward = np.std(rewards_plus + rewards_minus)
        episode_reward = (average_reward_plus + average_reward_minus) / 2
        episode_best_reward = max(max(rewards_plus), max(rewards_minus))

        # Update logs
        rewards_plus_logs.append(average_reward_plus)
        rewards_minus_logs.append(average_reward_minus)
        std_rewards_log.append(std_reward)
        episode_rewards.append(episode_reward)
        best_rewards_log.append(episode_best_reward if episode_best_reward > best_reward else best_reward)

        # Optional verbose logging
        if cfg.verbose:
            print(f"Episode {episode+1}:")
            print(f"Average Reward (Positive Perturbations): {average_reward_plus}")
            print(f"Average Reward (Negative Perturbations): {average_reward_minus}")
            print(f"Standard Deviation of Rewards: {std_reward}")
            print(f"Maximum Reward (Current Episode): {episode_best_reward}")
            # Update best reward
            if episode_best_reward > best_reward:
                best_reward = episode_best_reward
                print(f"New Best Reward: {best_reward}")
            print()

        ars_agent.update_policy(deltas, states_encountered, rewards_plus, rewards_minus)

    env_agent.gym_env.close()
    display_results(rewards_plus_logs, rewards_minus_logs, best_rewards_log, std_rewards_log, cfg)
