from typing import Tuple

import numpy as np
import openai
from overcookedgym.agents import StaticPolicyAgent
from stable_baselines3 import PPO

from LLMAPIs import GPT4API, DeepSeekR1API, LlamaAPI
from policy import PolicyModule


INTERACTION_WORDS = [
    "interact",
    "grab",
    "collect",
    "take",
    "place",
    "fill",
    "deliver",
    "pick up",
    "put",
    "serve",
    "drop",
    "add"]


class RandomAgent:
    def __init__(self):
        self.act_dim = 6

    def act(self, obs: np.ndarray) -> np.ndarray:
        action_idx = np.random.choice(6, size=(1,))
        action = np.zeros((6,))
        action[action_idx] = 1.0

        return action


class RLAgent:
    def __init__(self, checkpoint_path: str):
        agent = PPO.load(checkpoint_path)
        self.agent = StaticPolicyAgent(policy=agent.policy)

    def act(self, obs: Tuple[np.ndarray, str]) -> np.ndarray:
        obs_arr, obs_str = obs
        return self.agent.get_action(obs_arr)


class LLMAgent:
    def __init__(self, env, model_id: str = "gpt-4o-mini", context_length: int = 50, device: str = "cuda"):
        self.policy_model = PolicyModule(env)

        # if model_id == "gpt-4o-mini":
        if model_id.startswith("gpt-"):
            self.model_api = GPT4API(model_id)
        elif model_id.startswith("deepseek-ai"):
            self.model_api = DeepSeekR1API(model_id, device=device)
        elif model_id.startswith("meta-llama"):
            self.model_api = LlamaAPI(model_id, device=device)

        self.context_length = context_length

        # begin_info = "You are a chef in overcooked. Please cooperate with the other chef to make as many dishes as possible.\n"

        with open("game_rules.txt", "r") as f:
            game_rules = f.read()

        # additional_game_rules = """
        # The cramped room layout is a 5x4 grid of (x, y) with counter space at 
        # [(0, 0), (1, 0), (3, 0), (4, 0), (0, 2), (4, 2), (0, 3), (2, 3), (4, 3)], the pot at (2, 0), 
        # onion dispensers at [(0, 1), (4, 1)], the dish dispenser at (1, 3), and the serving location at (3, 3). 
        # The grid cells from (1, 1) down to (2, 3) are open kitchen space that the chefs can move around in. 
        # The chefs cannot move outside of this range as the remaining spaces are impassible counter space. 
        # You can interact with objects in adjacent grid cells if you are facing that direction e.g., 
        # if you are at cell (1, 1) and are facing left and an onion dispenser is at (0, 1) 
        # you can interact with it. If you are in cell (1, 1) and are facing up and the onion dispenser 
        # is at (0, 1), then you are adjacent but not facing the object, therefore you can not interact with it. 
        # You need to be adjacent to it and facing the object to interact with it. The first four actions 
        # correspond to moving to a new grid cell, each of which operate in an (x, y) coordinate system from the 
        # chefs perspective. **Up** moves the chef one cell upward on the y-axis e.g., moving up from grid cell 
        # (1, 2) would result in moving to grid cell (1, 1). **Down** moves the chef one cell down on the y-axis 
        # e.g., moving down from grid cell (1, 2) would result in moving to grid cell (1, 3). **Left** moves the 
        # chef one cell left on the x-axis e.g., moving left from grid cell (1, 2) would result in moving to grid 
        # cell (2, 2). **Right** moves the chef one cell right on the x-axis e.g., moving right from grid cell 
        # (1, 2) would result in moving to grid cell (1, 1). You must first get an onion from the dispenser, 
        # place three onions in the pot, cook the soup, then put the soup in a dish and serve the soup at the 
        # serving window. New dishes can be grabbed from the dish dispenser. If the chef is facing an empty tile, 
        # it can move to that tile. If the chef is facing a counter and are holding a dish, you can interact with 
        # it to place the dish on the counter. If the chef is facing either an onion dispenser, a dish dispenser, 
        # a pot, or a serving location, it can interact with it. If you are not facing an empty tile, you can not 
        # move into that cell. 
        # """

        self.system_message = [
            {
                "role": "system",
                # "content": begin_info + game_rules + additional_game_rules,
                "context": game_rules
            }
        ]

        self.history = []
        self.t = 0

    def act(self, obs: Tuple[np.ndarray, str]) -> np.ndarray:
        _, obs_str = obs

        game_info = (
            obs_str
            + " Take time to think through your strategy and enclose your thoughts inside <think></think>. Choose an action from your set of skills. Consider your past actions and how you should adapt your strategy to solve the task. Once you are done thinking, please output your action in following format: ###My action is {skill number}, without any other text."
        )
        self.history.append({"role": "user", "content": game_info})

        # valid_action = False
        # failure_message = None
        # while not valid_action:
        #     if failure_message:
        #         game_info = (
        #             obs_str
        #             + " Your last action was invalid. "
        #             + failure_message
        #             + " Please try again. "
        #             + " Take time to think through your strategy and enclose your thoughts inside <think></think>. Consider your past actions and how you should adapt your strategy to solve the task. Consider any dispensers or objects of interest that are adjacent to you and make sure you are facing them in order to interact with them. Ask yourself, 'If I am adjacent to an object, am I facing it?' If not, then you need to turn to face it to interact with it. Make sure to choose an action from either your movement skills or your cooking skills. Once you are done thinking, please output your action in following format: ###My action is {your action}, without any other text."
        #         )
        #         self.history[-1] = {"role": "user", "content": game_info}

        #     response = self.model_api.response(self.history[-self.context_length:])

        #     print("Response: ", response)

        #     action = self.response_to_action(response)

        #     if failure_message:
        #         print("Failure message: ", failure_message)
        #     print("Obs str: ", obs_str)
        #     print("Response: ", response)
        #     print("Action: ", action)

        #     valid_action, failure_message = self.validate_action(action, obs_str)

        response = self.model_api.response(self.history[-self.context_length:])

        action = self.response_to_action(response)

        actions = self.policy_model.get_action(action)
        print("Actions: ", actions)

        print("Obs str: ", obs_str)
        print("Response: ", response)
        print("Action: ", action)

        self.history.append({"role": "assistant", "content": response})

        # if self.t == 0:
        #     response = "stuff ### move up"
        # elif self.t == 1:
        #     response = "stuff ### move left"
        # elif self.t == 2:
        #     response = "stuff ### interact"
        # else:
        #     response = "stuff ### interact"

        # print("game_info: ", game_info)
        # print()
        # print("#########################################################")
        # print()
        # print("response: ", response)

        self.t += 1

        return action

    def response_to_action(self, response: str) -> np.ndarray:
        action_response = response.split("###")[1].lower()

        # if "up" in action_response:
        #     # print("Selecting action up.")
        #     action_idx = 0
        # elif "down" in action_response:
        #     # print("Selecting action down.")
        #     action_idx = 1
        # elif "right" in action_response:
        #     # print("Selecting action right.")
        #     action_idx = 2
        # elif "left" in action_response:
        #     # print("Selecting action left.")
        #     action_idx = 3
        # elif any(word in action_response for word in INTERACTION_WORDS):
        #     # print("Selecting action interact.")
        #     action_idx = 5
        # else:
        #     # print("Selecting action no-op.")
        #     action_idx = 4

        if "1" in action_response:
            action_idx = 0
        elif "2" in action_response:
            action_idx = 1
        elif "3" in action_response:
            action_idx = 2
        elif "4" in action_response:
            action_idx = 3
        elif "5" in action_response:
            action_idx = 4
        elif "6" in action_response:
            action_idx = 5
        elif "7" in action_response:
            action_idx = 6
        elif "8" in action_response:
            action_idx = 7
        elif "9" in action_response:
            action_idx = 8
        else:
            raise ValueError(f"Invalid action response: {action_response}")

        # action = np.zeros((6,))
        # action[action_idx] = 1.0

        return action_idx

    def validate_action(self, action: np.ndarray, obs_str: str) -> Tuple[bool, str]:
        action_idx = np.argmax(action)
        action_str = ["up", "down", "right", "left", "no-op", "interact"][action_idx]

        states = obs_str.split('\n')

        orientation_str = states[2]
        up_str = states[3]
        down_str = states[4]
        right_str = states[5]
        left_str = states[6]
        facing_str = states[7]
        held_obj_str = states[8]
        pot_state_str = states[9]

        print("Action: ", action_str)
        print("Orientation: ", orientation_str)
        print("Up: ", up_str)
        print("Down: ", down_str)
        print("Right: ", right_str)
        print("Left: ", left_str)
        print("Facing: ", facing_str)
        print("Held object: ", held_obj_str)
        print("Pot state: ", pot_state_str)

        if action_str == "no-op":
            return True, None

        if action_str == "up":
            if up_str == "- There is a counter above you. ":
                return False, "You cannot move up, there is a counter above you."
            else:
                return True, None
        elif action_str == "down":
            if down_str == "- There is a counter below you. ":
                return False, "You cannot move down, there is a counter below you."
            else:
                return True, None
        elif action_str == "left":
            if left_str == "- There is a counter to the left of you. ":
                return False, "You cannot move left, there is a counter to your left."
            else:
                return True, None
        elif action_str == "right":
            if right_str == "- There is an empty tile to the right of you. ":
                return False, "You cannot move right, there is a counter to your right."
            else:
                return True, None
        elif action_str == "interact":
            if facing_str == "- You are facing the counter. ":
                return False, "You cannot interact, you are facing a counter."


            # Edge cases for interaction:

            # Holding nothing:
            # - If you are holding nothing and facing a counter, you cannot interact.

            # Holding on onion:
            # - If you are holding an onion and are not facing a pot, you cannot interact.
            # - If you are holding an onion and facing an onion dispenser, you cannot interact.
            # - If you are holding an onion and facing a dish dispenser, you cannot pick up a dish.
            # - If you are holding an onion and facing a serving location, you cannot serve the onion.
            # - If you are holding an onion and facing a pot, you cannot interact with the pot if the pot is cooking or ready to serve.

            # Holding a dish:
            # - If you are holding a dish and facing a dish dispenser, you cannot pick up a dish.
            # - If you are holding a dish and facing a pot, you cannot interact with the pot unless it is ready to serve.
            # - If you are holding a dish and facing a serving location, you can serve the dish.
            # - If you are holding a dish and facing a counter, you can place the dish on the counter.
            # - If you are holding a dish and facing an onion dispenser, you cannot interact with the onion dispenser.

            if held_obj_str == "- You are holding nothing. ":
                if facing_str == "- You are facing a counter. ":
                    return False, "You cannot interact, you are holding nothing and facing a counter."
                else:
                    return True, None

            elif held_obj_str == "- You are holding an onion. ":
                if facing_str == "- You are facing a pot. ":
                    if pot_state_str == "- The pot is cooking. ":
                        return False, "You cannot interact with the pot, it is cooking."
                    elif pot_state_str == "- The pot is ready to serve. ":
                        return False, "You cannot interact with the pot, it is ready to serve."
                    else:
                        return True, None
                elif facing_str == "- You are facing an onion dispenser. ":
                    return False, "You cannot interact with the onion dispenser, you are holding an onion."
                elif facing_str == "- You are facing a dish dispenser. ":
                    return False, "You cannot pick up a dish, you are holding an onion."
                elif facing_str == "- You are facing a serving location. ":
                    return False, "You cannot serve the onion, you are holding an onion."
                else:
                    return True, None

            elif held_obj_str == "- You are holding a dish. ":
                if facing_str == "- You are facing a dish dispenser. ":
                    return False, "You cannot pick up a dish, you are holding a dish."
                elif facing_str == "- You are facing a pot. ":
                    if pot_state_str == "- The pot is cooking. ":
                        return False, "You cannot interact with the pot, it is cooking."
                    elif pot_state_str == "- The pot is ready to serve. ":
                        return True, None
                    else:
                        return True, None
                elif facing_str == "- You are facing a serving location. ":
                    return True, None
                elif facing_str == "- You are facing a counter. ":
                    return True, None
                elif facing_str == "- You are facing an onion dispenser. ":
                    return False, "You cannot interact with the onion dispenser, you are holding a dish."
                else:
                    return True, None
            else:
                return True, None
