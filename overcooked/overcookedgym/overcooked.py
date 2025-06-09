import gym
import numpy as np
from pprint import pprint
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from .multiagentenv import SimultaneousEnv


DIRECTION_TO_STR = {0: "up", 1: "down", 2: "right", 3: "left"}
ACTION_TO_STR = {
    0: "move up",
    1: "move down",
    2: "move right",
    3: "move left",
    4: "stay",
    5: "interact",
}


class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        DEFAULT_ENV_PARAMS = {"horizon": 400}
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name=layout_name, rew_shaping_params=rew_shaping_params
        )
        self.mlp = MediumLevelPlanner.from_pickle_or_compute(
            self.mdp, NO_COUNTERS_PARAMS, force_compute=False
        )

        self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, self.mlp)
        self.describe_fn = lambda x: self.describe_state(x, self.mlp)

        if baselines:
            np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(self.lA)
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = (
            np.ones(obs_shape, dtype=np.float32) * np.inf
        )  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(-high, high, dtype=np.float64)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = (
            Action.INDEX_TO_ACTION[ego_action],
            Action.INDEX_TO_ACTION[alt_action],
        )
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info["shaped_r"]
        reward = reward + rew_shape

        # print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs), (reward, reward), done, {}  # info

    def step(self, actions):
        if self.ego_agent_idx == 0:
            ego_action = actions[0]
            alt_action = actions[1]
        else:
            ego_action = actions[1]
            alt_action = actions[0]

        ego_action_idx = ego_action.argmax()
        alt_action_idx = alt_action.argmax()

        ego_action, alt_action = (
            Action.INDEX_TO_ACTION[ego_action_idx],
            Action.INDEX_TO_ACTION[alt_action_idx],
        )
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
            joint_action_idx = (ego_action_idx, alt_action_idx)
        else:
            joint_action = (alt_action, ego_action)
            joint_action_idx = (alt_action_idx, ego_action_idx)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info["shaped_r"]
        reward = reward + rew_shape

        # print(self.base_env.mdp.state_string(next_state))
        ob_p0_str, ob_p1_str = self.describe_fn((self.base_env.state, joint_action_idx))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        obs = [(ob_p0.reshape(1, -1), ob_p0_str), (ob_p1.reshape(1, -1), ob_p1_str)]

        rewards = np.array([reward, reward])
        dones = np.array([done, done]).astype(np.bool_)

        return obs, rewards, dones, info

    def reset(self):
        self.base_env.reset()
        ob_p0_str, ob_p1_str = self.describe_fn((self.base_env.state, None))
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        obs = [(ob_p0.reshape(1, -1), ob_p0_str), (ob_p1.reshape(1, -1), ob_p1_str)]

        return obs

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)

    def render(self, mode="human", close=False):
        pass

    def describe_state(self, inputs, mlp) -> str:
        # TODO: Add information for onion dispenser location, pot location, pot status etc.

        state, actions = inputs

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features[
                "p{}_closest_{}".format(idx, name)
            ] = self.mdp.get_deltas_to_closest_location(player, locations, mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.mdp.get_counter_objects_dict(state)
        pot_state = self.mdp.get_pot_states(state)

        # Player Info
        player_states = []
        for i, player in enumerate(state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            player_orientation_str = DIRECTION_TO_STR[orientation_idx]
            position = player.position
            # orientation_str = "Chef {} is standing in cell {}. ".format(i, position)
            orientation_str = (
                "You are standing in cell {} and are facing {}. ".format(
                    position, player_orientation_str
                )
            )

            obj = player.held_object
            if obj is None:
                held_obj_name = "none"
                held_obj_str = "You are holding nothing. "
            else:
                held_obj_name = obj.name
                held_obj_str = "You are holding a {}".format(obj.name)

            # Closest feature of each type
            if held_obj_name == "onion":
                # print("Holding onion")
                held_obj_str = "You are holding an onion. "

            dispenser_locations = self.mdp.get_onion_dispenser_locations()
            dispenser_locations_str = f"There are onion dispensers located in cell {dispenser_locations[0]} and cell {dispenser_locations[1]}. "

            if len(counter_objects["onion"]) == 0:
                onion_locations_str = "There are no onions already on the counter. "
            elif len(counter_objects["onion"]) == 1:
                onion_location = counter_objects["onion"][0]
                onion_locations_str = f"There is an onion on the counter located at cell {onion_location}. "
            else:
                onion_locations_str = "There are onions on the counter located at cell "
                for j in range(len(counter_objects["onion"]) - 1):
                    onion_location = counter_objects["onion"][j]
                    onion_locations_str += f"{onion_location} and cell "
                onion_location = counter_objects["onion"][j + 1]
                onion_locations_str += f"{onion_location}. "

                # print(onion_locations_str)

            if len(pot_state["empty"]) == 1:
                empty_pot_location = pot_state["empty"][0]
                empty_pot_str = f"There is an empty pot at cell {empty_pot_location}. "
                pot_state_str = "The pot is empty. "
            else:
                empty_pot_str = ""

            if len(pot_state["onion"]["partially_full"]) == 1:
                pot_state_str = "The pot is partially full. "

            if len(pot_state["onion"]["one_onion"]) == 1:
                one_onion_pot_location = pot_state["onion"]["one_onion"][0]
                one_onion_pot_str = f"There is a pot with one onion in it already at cell {one_onion_pot_location}. "
                pot_state_str = "There is a pot with one onion in it. "
            else:
                one_onion_pot_str = ""

            if len(pot_state["onion"]["two_onion"]) == 1:
                two_onion_pot_location = pot_state["onion"]["two_onion"][0]
                two_onion_pot_str = f"There is a pot with two onions in it already at cell {two_onion_pot_location}. "
                pot_state_str = "There is a pot with two onions in it. "
            else:
                two_onion_pot_str = ""

            if len(pot_state["onion"]["cooking"]) == 1:
                cooking_pot_location = pot_state["onion"]["cooking"][0]
                cooking_pot_str = (
                    f"Onion soup is cooking in the pot at cell {cooking_pot_location}. "
                )
                pot_state_str = "The pot is cooking onion soup. "
            else:
                cooking_pot_str = "There is no onion soup cooking at the moment. "

            if len(pot_state["onion"]["ready"]) == 1:
                ready_pot_location = pot_state["onion"]["ready"][0]
                ready_pot_str = f"There is a pot of cooked soup ready to serve at cell {ready_pot_location}. "
                pot_state_str = "The pot is ready to serve onion soup. "
            else:
                ready_pot_str = "There is no onion soup ready to serve at the moment. "

            if held_obj_name == "dish":
                # print("Get closest onion location.")
                held_obj_str = "You are holding a dish. "

            dish_dispenser_location = self.mdp.get_dish_dispenser_locations()
            dish_dispenser_location_str = f"There is a dish dispenser located in cell {dish_dispenser_location[0]}. "

            if len(counter_objects["dish"]) == 0:
                dish_locations_str = "There are no dishes already on the counter. "
            elif len(counter_objects["dish"]) == 1:
                dish_location = counter_objects["dish"][0]
                dish_locations_str = (
                    f"There is a dish on the counter located at cell {dish_location}. "
                )
            else:
                dish_locations_str = "There are dishes on the counter located at cell "
                for j in range(len(counter_objects["dish"]) - 1):
                    dish_location = counter_objects["dish"][j]
                    dish_locations_str += f"{dish_location} and cell "
                dish_location = counter_objects["dish"][j + 1]
                dish_locations_str += f"{dish_location}. "

            if held_obj_name == "soup":
                held_obj_str = "You are holding an onion soup. "

            if len(counter_objects["soup"]) == 0:
                soup_locations_str = "There are no soups already on the counter. "
            elif len(counter_objects["soup"]) == 1:
                soup_location = counter_objects["soup"][0]
                soup_locations_str = (
                    f"There is a soup on the counter located at cell {soup_location}. "
                )
            else:
                soup_locations_str = "There are soups on the counter located at cell "
                for j in range(len(counter_objects["soup"]) - 1):
                    soup_location = counter_objects["soup"][j]
                    soup_locations_str += f"{soup_location} and cell "
                soup_location = counter_objects["soup"][j + 1]
                soup_locations_str += f"{soup_location}. "

            serving_location = self.mdp.get_serving_locations()[0]
            serving_location_str = (
                f"The serving window is located at cell {serving_location}. "
            )
            # print(serving_location_str)

            for direction_idx, pos_and_feat in enumerate(
                self.mdp.get_adjacent_features(player)
            ):
                direction_str = DIRECTION_TO_STR[direction_idx]
                adj_pos, feat = pos_and_feat

                # print("Chef {} position: {}".format(i, player.position))
                # print("Chef {} direction: {}".format(i, player_orientation_str))
                # print(
                #     "Chef {} adjacent features direction: {}".format(i, direction_str)
                # )
                # print("Chef {} adjacent positions: {}".format(i, adj_pos))
                # print("Chef {} adjacent features: {}".format(i, feat))

                if direction_str == player_orientation_str:
                    if feat == "X":
                        facing_str = "You are facing the counter. "
                    elif feat == "O":
                        facing_str = "You are facing an onion dispenser. "
                    elif feat == "P":
                        facing_str = "You are facing a pot. "
                    elif feat == "D":
                        facing_str = "You are facing a dish dispenser. "
                    elif feat == "S":
                        facing_str = "You are facing a serving location. "
                    else:
                        facing_str = "You are facing an empty tile. "

                if direction_str == "up":
                    if feat == "X":
                        up_str = "There is a counter above you. "
                    elif feat == "O":
                        up_str = "There is an onion dispenser above you. "
                    elif feat == "P":
                        up_str = "There is a pot above you. "
                    elif feat == "D":
                        up_str = "There is a dish dispenser above you. "
                    elif feat == "S":
                        up_str = "There is a serving location above you. "
                    else:
                        up_str = "There is an empty tile above you. "

                if direction_str == "down":
                    if feat == "X":
                        down_str = "There is a counter below you. "
                    elif feat == "O":
                        down_str = "There is an onion dispenser below you. "
                    elif feat == "P":
                        down_str = "There is a pot below you. "
                    elif feat == "D":
                        down_str = "There is a dish dispenser below you. "
                    elif feat == "S":
                        down_str = "There is a serving location below you. "
                    else:
                        down_str = "There is an empty tile below you. "

                if direction_str == "right":
                    if feat == "X":
                        right_str = "There is a counter to the right of you. "
                    elif feat == "O":
                        right_str = "There is an onion dispenser to the right of you. "
                    elif feat == "P":
                        right_str = "There is a pot to the right of you. "
                    elif feat == "D":
                        right_str = "There is a dish dispenser to the right of you. "
                    elif feat == "S":
                        right_str = "There is a serving location to the right of you. "
                    else:
                        right_str = "There is an empty tile to the right of you. "

                if direction_str == "left":
                    if feat == "X":
                        left_str = "There is a counter to the left of you. "
                    elif feat == "O":
                        left_str = "There is an onion dispenser to the left of you. "
                    elif feat == "P":
                        left_str = "There is a pot to the left of you. "
                    elif feat == "D":
                        left_str = "There is a dish dispenser to the left of you. "
                    elif feat == "S":
                        left_str = "There is a serving location to the left of you. "
                    else:
                        left_str = "There is an empty tile to the left of you. "

            # player_str = (
            #     orientation_str
            #     + up_str
            #     + down_str
            #     + right_str
            #     + left_str
            #     + facing_str
            #     + held_obj_str
            #     + dispenser_locations_str
            #     + onion_locations_str
            #     + empty_pot_str
            #     + one_onion_pot_str
            #     + two_onion_pot_str
            #     + cooking_pot_str
            #     + ready_pot_str
            #     + dish_dispenser_location_str
            #     + dish_locations_str
            #     + soup_locations_str
            #     + serving_location_str
            # )

            try:
                player_str = f"\nChef {i} state:\n- {orientation_str}\n- {up_str}\n- {down_str}\n- {right_str}\n- {left_str}\n- {facing_str}\n- {held_obj_str}\n- {pot_state_str}\n"
            except Exception as e:
                print(pot_state)
                raise e

            player_states.append(player_str)

        p0_state_str = player_states[0]
        p1_state_str = player_states[1]

        if actions:
            p0_state_str += "In the last timestep, you performed action: {}. In the last timestep the other chef performed the action: {}.".format(
                ACTION_TO_STR[actions[0]],
                ACTION_TO_STR[actions[1]],
            )

            p1_state_str += "In the last timestep, you performed action: {}. In the last timestep the other chef performed the action: {}.".format(
                ACTION_TO_STR[actions[1]],
                ACTION_TO_STR[actions[0]],
            )

        state_strings = [p0_state_str, p1_state_str]

        return state_strings
