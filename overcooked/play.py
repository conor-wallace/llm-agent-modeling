import gym
from overcookedgym.overcooked_utils import LAYOUT_LIST
from agents import RLAgent, RandomAgent, LLMAgent
from policy import PolicyModule
from tqdm import tqdm


def make_env():
    env_config = {'layout_name': 'simple'}
    env = gym.make('OvercookedMultiEnv-v0', **env_config)

    return env


def play():

    env = make_env()
    obs = env.reset()

    ego_agent = RLAgent('checkpoints/OvercookedMultiEnv-v0-simple-PPO-ego-10.zip')
    # ego_agent = LLMAgent(model_id="gpt-4o-mini", context_length=50, device="cuda")

    # alt_agent = RLAgent('checkpoints/OvercookedMultiEnv-v0-simple-PPO-alt-10.zip')
    alt_agent = LLMAgent(env, model_id="gpt-4o-mini", context_length=50, device="cuda")

    # pm = PolicyModule(env)

    # exit()

    episode_reward = 0
    episode_length = 1
    for step in tqdm(range(episode_length), desc="Episode", total=episode_length):
        ego_agent_action = ego_agent.act(obs[0])  # Get action from the ego agent
        alt_agent_action = alt_agent.act(obs[1])  # Get action from the alternative agent

        action = [ego_agent_action, alt_agent_action]  # Combine actions for both agents
        obs, reward, done, info = env.step(action)

        episode_reward += reward

    print(f"Episode reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    play()