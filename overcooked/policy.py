from overcooked_ai_py.planning.planners import HighLevelActionManager


class PolicyModule:
    def __init__(self, env):
        self.env = env
        self.action_manager = env.unwrapped.mlp.ml_action_manager

    def get_action(self, action_idx: int):
        state = self.env.unwrapped.base_env.state
        counter_objects = self.env.unwrapped.mdp.get_counter_objects_dict(state)
        pot_state = self.env.unwrapped.mdp.get_pot_states(state)

        player = state.players[1]

        if action_idx == 0:
            return self.action_manager.pickup_onion_actions(state, counter_objects)
        elif action_idx == 1:
            return self.action_manager.pickup_dish_actions(state, counter_objects)
        elif action_idx == 2:
            return self.action_manager.pickup_counter_soup_actions(state, counter_objects)
        elif action_idx == 3:
            return self.action_manager.place_obj_on_counter_actions(state)
        elif action_idx == 4:
            return self.action_manager.deliver_soup_actions()
        elif action_idx == 5:
            return self.action_manager.put_onion_in_pot_actions(pot_state)
        elif action_idx == 6:
            return self.action_manager.pickup_soup_with_dish_actions(pot_state)
        elif action_idx == 7:
            return self.action_manager.go_to_closest_feature_actions(player)
        elif action_idx == 8:
            return self.action_manager.wait_actions(player)