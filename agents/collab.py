import json

from langchain.chat_models import init_chat_model
from langchain.schema.messages import SystemMessage

from .base import BaseAgent
from .features import fingerprint_from_window
from .history import TeammateHistory
from .obs import describe_overcooked_obs_enhanced
from .prompts import SYSTEM_PROMPT, TASK_PROMPT


class CoLLABAgent(BaseAgent):
    def __init__(
        self, config, env, checkpoint_dir, team_dirs, agent_idx
    ):
        super().__init__(config, env, checkpoint_dir, team_dirs["DEFAULT"], agent_idx)

        self.t = 0
        self.team_dirs = team_dirs
        self.pred_teammate_type = "DEFAULT"
        self.hist = TeammateHistory(K=self.config["K"])
        self.model = init_chat_model(self.config["MODEL_ID"])

    def act(self, obs, prev_actions, prev_rewards):
        obs_0 = obs["agent_0"]
        obs_1 = obs["agent_1"]
        prev_action_0 = prev_actions["agent_0"]
        prev_action_1 = prev_actions["agent_1"]
        prev_reward_0 = prev_rewards["agent_0"]
        prev_reward_1 = prev_rewards["agent_1"]

        self.hist.push(
            obs_0,
            obs_1,
            prev_action_0,
            prev_action_1,
            prev_reward_0,
            prev_reward_1
        )

        if self.hist.loaded and self.t <= self.config["K"]:
            agent_features_dict_0 = fingerprint_from_window(
                self.hist.agent_0_obs,
                self.hist.agent_0_actions,
                self.hist.agent_0_rewards
            )

            task_prompt = TASK_PROMPT.format(
                cumulative_reward=f"{agent_features_dict_0["cumulative_reward"]:.2f}",
                dwell_onion=f"{agent_features_dict_0["dwell_onion"]:.2f}",
                dwell_plate=f"{agent_features_dict_0["dwell_plate"]:.2f}",
                dwell_pot=f"{agent_features_dict_0["dwell_pot"]:.2f}",
                dwell_window=f"{agent_features_dict_0["dwell_window"]:.2f}",
                near_onion_pile_steps=f"{agent_features_dict_0["near_onion_pile_steps"]:.2f}",
                near_plate_pile_steps=f"{agent_features_dict_0["near_plate_pile_steps"]:.2f}",
                near_pot_steps=f"{agent_features_dict_0["near_pot_steps"]:.2f}",
                near_window_steps=f"{agent_features_dict_0["near_window_steps"]:.2f}"
            )

            print(task_prompt)

            response = self.model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    SystemMessage(content=task_prompt)
                ]
            )

            print(response.content)

            self.pred_teammate_type = json.loads(response.content)["teammate_type"]
            self.team_dir = self.team_dirs.get(self.pred_teammate_type)
            self.load_params()

        self.t += 1

        obs_i = obs[f"agent_{self.agent_idx}"].flatten()
        pi, _ = self.network.apply(self.network_params, obs_i)
        return pi