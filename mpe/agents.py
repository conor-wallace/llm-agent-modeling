from typing import Tuple

import numpy as np
import torch
from models import PPONet
from LLMAPIs import GPT4API


def after_first_landmark(obs):
    agent_obs = obs[4:6]

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_second_landmark(obs):
    agent_obs = obs[6:8]

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_third_landmark(obs):
    agent_obs = obs[8:10]

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_farthest_landmark(obs):
    landmark1_rel_pos = obs[4:6]
    landmark2_rel_pos = obs[6:8]
    landmark3_rel_pos = obs[8:10]

    landmark1_dist = (landmark1_rel_pos**2).sum()
    landmark2_dist = (landmark2_rel_pos**2).sum()
    landmark3_dist = (landmark3_rel_pos**2).sum()
    dists = np.array([landmark1_dist, landmark2_dist, landmark3_dist])
    index = np.argmax(dists)
    if index == 0:
        agent_obs = landmark1_rel_pos
    elif index == 1:
        agent_obs = landmark2_rel_pos
    else:
        agent_obs = landmark3_rel_pos

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_closest_landmark(obs):
    landmark1_rel_pos = obs[4:6]
    landmark2_rel_pos = obs[6:8]
    landmark3_rel_pos = obs[8:10]

    landmark1_dist = (landmark1_rel_pos**2).sum()
    landmark2_dist = (landmark2_rel_pos**2).sum()
    landmark3_dist = (landmark3_rel_pos**2).sum()
    dists = np.array([landmark1_dist, landmark2_dist, landmark3_dist])
    index = np.argmin(dists)
    if index == 0:
        agent_obs = landmark1_rel_pos
    elif index == 1:
        agent_obs = landmark2_rel_pos
    else:
        agent_obs = landmark3_rel_pos

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


class RandomAgent:
    def __init__(self):
        self.act_dim = 5

    def act(self, obs: np.ndarray) -> np.ndarray:
        action_idx = np.random.choice(5, size=(1,))
        action = np.zeros((5,))
        action[action_idx] = 1.0

        return action


class RLAgent:
    def __init__(self, checkpoint_path: str):
        self.agent = PPONet(18, 5, 64)
        save_dict = torch.load(checkpoint_path)
        modified_save_dict = {
            'base.feature_norm.weight': save_dict['base.feature_norm.weight'],
            'base.feature_norm.bias': save_dict['base.feature_norm.bias'],
            'base.mlp.fc1.0.bias': save_dict['base.mlp.fc1.0.bias'],
            'base.mlp.fc1.0.weight': save_dict['base.mlp.fc1.0.weight'],
            'base.mlp.fc1.2.weight': save_dict['base.mlp.fc1.2.weight'],
            'base.mlp.fc1.2.bias': save_dict['base.mlp.fc1.2.bias'],
            'base.mlp.fc2.0.weight': save_dict['base.mlp.fc2.0.0.weight'],
            'base.mlp.fc2.0.bias': save_dict['base.mlp.fc2.0.0.bias'],
            'base.mlp.fc2.2.weight': save_dict['base.mlp.fc2.0.2.weight'],
            'base.mlp.fc2.2.bias': save_dict['base.mlp.fc2.0.2.bias'],
            'act.weight': save_dict['act.action_out.linear.weight'],
            'act.bias': save_dict['act.action_out.linear.bias']
        }
        self.agent.load_state_dict(modified_save_dict)
        self.agent.eval()

    def act(self, obs: Tuple[np.ndarray, str]) -> np.ndarray:
        obs_arr, _, _ = obs
        return self.agent(obs_arr)


class StaticAgent:
    def __init__(self, policy: int = 0):
        self.policy = policy

        self.policy_mapping = {
            0: after_first_landmark,
            1: after_second_landmark,
            2: after_third_landmark,
            3: after_closest_landmark,
            4: after_farthest_landmark
        }

    def act(self, obs: Tuple[np.ndarray, str]) -> np.ndarray:
        obs_arr, _, _ = obs
        action = self.policy_mapping[self.policy](obs_arr)
        return action


class LLMAgent:
    def __init__(self, model_id: str = "gpt-4o-mini", context_length: int = 50):
        if model_id.startswith("gpt-"):
            self.model_api = GPT4API(model_id)

        with open("game_rules.txt", "r") as f:
            game_rules = f.read()

        self.history = [
            {
                "role": "system",
                "content": game_rules
            }
        ]
        self.context_length = context_length

    def act(self, obs: Tuple[np.ndarray, str]) -> np.ndarray:
        _, obs_str, reward = obs



        valid_action = False
        failure_message = None
        while not valid_action:
            game_info = (
                obs_str
                # + " The reward you achieved from your previous action was " + str(reward)
                + " Take time to think through your strategy and enclose your thoughts inside <think></think>. Choose an action from your set of skills. Consider your past actions and how you should adapt your strategy to solve the task. Once you are done thinking, please output your action in following format: ###My action is {skill number}, without any other text."
            )
            self.history.append({"role": "user", "content": game_info})

            response = self.model_api.response(self.history[-self.context_length:])

            try:
                action = self.response_to_action(response)
                valid_action = True
            except Exception as e:
                failure_message = e
                valid_action = False

        return action

    def response_to_action(self, response: str) -> np.ndarray:
        action_response = response.split("###")[1].lower()

        if '1' in action_response:
            action_idx = 0
        elif '2' in action_response:
            action_idx = 1
        elif '3' in action_response:
            action_idx = 2
        elif '4' in action_response:
            action_idx = 3
        elif '5' in action_response:
            action_idx = 4
        else:
            raise ValueError(f"Invalid action response: {action_response}")

        action = np.zeros(5)
        action[action_idx] = 1

        return action