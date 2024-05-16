import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from functools import partial
from moviepy.editor import ipython_display as video_display

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env, record_video

from ARSAgent_v1 import ARSAgent_v1
from ARSAgent_v2 import ARSAgent_v2
from Logger import Logger

# Create the environment Agent
def create_env_agent(cfg):
    """
    Create and initialize a parallel Gym environment agent.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        ParallelGymAgent: Initialized parallel Gym environment agent.
    """
    env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name),
        cfg.algorithm.n_envs,
        ).seed(cfg.algorithm.seed)
    return env_agent


# Create the ARS Agent
def create_ARS_agent(cfg, env_agent):
    """
    Create and initialize an ARS (Augmented Random Search) agent.

    Args:
        cfg (DictConfig): Configuration dictionary.
        env_agent (ParallelGymAgent): Parallel Gym environment agent.

    Returns:
        ARSAgent_v1 or ARSAgent_v2: Initialized ARS agent based on configuration.
    """
    observation_dim, action_dim = env_agent.get_obs_and_actions_sizes()
    M = np.zeros((action_dim, observation_dim))

    if cfg.algorithm.v2:
        ars_agent = ARSAgent_v2(env_agent, cfg, M=M)
    else:
        ars_agent = ARSAgent_v1(env_agent, cfg, M=M)

    return ars_agent


def init_agents(cfg):
    """
    Initialize environment agent, ARS agent, and evaluation agent.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        tuple: Tuple containing environment agent, ARS agent, and evaluation agent.
    """
    env_agent = create_env_agent(cfg)
    ars_agent = create_ARS_agent(cfg, env_agent)
    composed_agent = Agents(env_agent, ars_agent)
    eval_agent = TemporalAgent(composed_agent)
    
    return env_agent, ars_agent, eval_agent


def run_episode_with_perturbation(ars_agent, delta, eval_agent):
    """
    Run an episode with perturbation applied to the ARS agent.

    Args:
        ars_agent (ARSAgent): ARS agent instance.
        delta (numpy.ndarray): Perturbation vector.
        eval_agent (TemporalAgent): Evaluation agent.

    Returns:
        tuple: Tuple containing episode rewards, states encountered, and number of steps.
    """
	ars_agent.set_delta(delta)
	workspace = Workspace()
	eval_agent(workspace, t=0, stop_variable="env/done")
	rewards = workspace["env/cumulated_reward"][-1]
	states = workspace["env/env_obs"][0].tolist()
	nb_steps = workspace["action"].shape[0]
	ars_agent.reset_delta()

	return rewards, states, nb_steps


def display_results(rewards_plus_logs, rewards_minus_logs, best_rewards_log, std_rewards_log, cfg):
	"""
    Display ARS performance results across episodes.

    Args:
        rewards_plus_logs (list): List of average rewards for positive perturbations.
        rewards_minus_logs (list): List of average rewards for negative perturbations.
        best_rewards_log (list): List of best rewards achieved.
        std_rewards_log (list): List of standard deviations of rewards.
        cfg (DictConfig): Configuration dictionary.
    """
    plt.figure(figsize=(10, 6))
    episodes = range(cfg.algorithm.num_episodes)
    
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
    """
    Run the Augmented Random Search (ARS) algorithm.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
	logger = Logger(cfg)
	env_agent, ars_agent, eval_agent = init_agents(cfg)
	
	best_reward = -np.inf
	rewards_plus_logs = []
	rewards_minus_logs = []
	best_rewards_log = []
	std_rewards_log = []
	episode_rewards = []

	nb_steps_rmax = 0
	nb_steps_rmin = 0

    # Iterate over episodes
	for episode in range(cfg.algorithm.num_episodes):
		# Generate random perturbations
		deltas = np.random.randn(ars_agent.N, *ars_agent.M.shape)
		rewards_plus = []     # To store rewards when adding perturbations
		rewards_minus = []    # To store rewards when subtracting perturbations
		states_encountered = []

        # Evaluate perturbations
		for delta in deltas:
			rewards, states, nb_steps = run_episode_with_perturbation(ars_agent, delta, eval_agent)
			rewards_plus.append(rewards.item())
			states_encountered.extend(states)
			nb_steps_rmax += nb_steps
			logger.add_log("reward_max", rewards, nb_steps_rmax)
			
			rewards, states, nb_steps = run_episode_with_perturbation(ars_agent, -delta, eval_agent)
			rewards_minus.append(rewards.item())
			states_encountered.extend(states)
			nb_steps_rmin += nb_steps
			logger.add_log("reward_min", rewards, nb_steps_rmin)

		# Calculate statistics for logging
		average_reward_plus = np.mean(rewards_plus)
		average_reward_minus = np.mean(rewards_minus)
		std_reward = np.std(rewards_plus + rewards_minus)
		episode_reward = (average_reward_plus + average_reward_minus) / 2
		episode_best_reward = max(max(rewards_plus), max(rewards_minus))
		
		# Update best reward
		if episode_best_reward > best_reward:
			best_reward = episode_best_reward
		
		# Update logs
		rewards_plus_logs.append(average_reward_plus)
		rewards_minus_logs.append(average_reward_minus)
		std_rewards_log.append(std_reward)
		episode_rewards.append(episode_reward)
		best_rewards_log.append(best_reward)

		# Optional verbose logging
		if cfg.verbose:
			print(f"Episode {episode+1}:")
			print(f"Average Reward (Positive Perturbations): {average_reward_plus}")
			print(f"Average Reward (Negative Perturbations): {average_reward_minus}")
			print(f"Standard Deviation of Rewards: {std_reward}")
			print(f"Maximum Reward (Current Episode): {episode_best_reward}")
			if episode_best_reward == best_reward:
				print(f"New Best Reward: {best_reward}")
			print()

        # Update ARS agent's policy based on perturbation results
		ars_agent.update_policy(deltas, states_encountered, rewards_plus, rewards_minus)

	display_results(rewards_plus_logs, rewards_minus_logs, best_rewards_log, std_rewards_log, cfg)
	
	# Optionally render and display video of the best policy
	if cfg.render:
		version = "v2" if cfg.algorithm.v2 else "v1"
		env = make_env(cfg.gym_env.env_name, render_mode="rgb_array")
		record_video(env, ars_agent, f"videos/{cfg.gym_env.env_name}_ARS_{version}.mp4")
		video_display(f"videos/{cfg.gym_env.env_name}_ARS_{version}.mp4")
