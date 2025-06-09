import os
import gym
import multiagent.scenarios as scenarios
import numpy as np
from multiagent.environment import MultiAgentEnv
from tqdm import tqdm

from agents import RLAgent, RandomAgent, StaticAgent, LLMAgent
from utils import describe_observation, save_trajectory


def make_env():
    scenario = scenarios.load("simple_spread.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        discrete_action=True
    )
    return env


def play():
    env = make_env()
    env._seed(0)

    # RL Agent: -63.26612441096736 +- 7.839555925086319
    # Random Agent: -50.40039707212408 +- 5.749418039241855
    # Static Agent 1: -60.30920019249611 +- 6.270108419770015
    # Static Agent 2: -46.62807030594947 +- 12.140580419139203
    # Static Agent 3: -67.54261162312226 +- 7.676412485606621
    # Static Agent 4: -53.81863196173939 +- 18.303833721609575
    # Static Agent 5: -57.42195753480869 +- 12.251444924625483
    # LLM Agent: -50.881046765552 +- 8.71169114408052
    # LLM Agent + Reflection: -48.243958243934216 +- 7.20465900072035

    save_dir = "trajectories/spread/static_2"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    avg_rewards = []
    num_iterations = 3
    for i in range(num_iterations):
        obs = env.reset()

        agent_1 = RLAgent('checkpoints/spread/params_mappo_agent_0.pt')
        # agent_1 = RandomAgent()

        agent_2 = RLAgent('checkpoints/spread/params_mappo_agent_1.pt')
        # agent_2 = RandomAgent()

        # agent_3 = RLAgent('checkpoints/spread/params_mappo_agent_2.pt')
        # agent_3 = RandomAgent()
        # agent_3 = StaticAgent(policy=0)
        # agent_3 = StaticAgent(policy=1)
        agent_3 = StaticAgent(policy=2)
        # agent_3 = StaticAgent(policy=3)
        # agent_3 = StaticAgent(policy=4)
        # agent_3 = LLMAgent()

        agent_1_reward = 0
        agent_2_reward = 0
        agent_3_reward = 0
        episode_reward = 0
        episode_length = 25
        trajectory = []
        for step in tqdm(range(episode_length), desc="Episode", total=episode_length):
            obs_1_desc = describe_observation(obs[0])
            obs_2_desc = describe_observation(obs[1])
            obs_3_desc = describe_observation(obs[2])
            agent_1_action = agent_1.act((obs[0], obs_1_desc, agent_1_reward))  # Get action from the 1st agent
            agent_2_action = agent_2.act((obs[1], obs_2_desc, agent_2_reward))  # Get action from the 2nd agent
            agent_3_action = agent_3.act((obs[2], obs_3_desc, agent_3_reward))  # Get action from the 3rd agent

            action = [agent_1_action, agent_2_action, agent_3_action]  # Combine actions for all agents
            obs, reward, done, info = env.step(action)

            trajectory.append(obs_3_desc)

            agent_1_reward = reward[0]
            agent_2_reward = reward[1]
            agent_3_reward = reward[2]

            episode_reward += agent_3_reward

        avg_rewards.append(episode_reward)

        save_trajectory(os.path.join(save_dir, f"trajectory_{i}.txt"), trajectory)

    reward_avg = np.mean(avg_rewards)
    reward_std = np.std(avg_rewards)

    print(f"Average reward: {reward_avg}, standard deviation: {reward_std}")

    env.close()


if __name__ == "__main__":
    play()