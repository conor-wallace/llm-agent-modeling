import random

from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(
        self, config, env, checkpoint_dir, team_dirs, agent_idx
    ):
        self.team_dirs = team_dirs
        self.pred_teammate_type = "DEFAULT"

        super().__init__(config, env, checkpoint_dir, team_dirs["DEFAULT"], agent_idx)

    def act(self, obs, prev_actions, prev_rewards):
        self.pred_teammate_type = random.choice(list(self.team_dirs.keys()))
        self.team_dir = self.team_dirs.get(self.pred_teammate_type)
        self.load_params()

        return super().act(obs, prev_actions, prev_rewards)