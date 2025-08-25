from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TeammateHistory:
    K: int = 10
    loaded: bool = False
    agent_0_obs: List[np.ndarray] = field(default_factory=list)
    agent_1_obs: List[np.ndarray] = field(default_factory=list)
    agent_0_actions: List[int] = field(default_factory=list)
    agent_1_actions: List[int] = field(default_factory=list)
    agent_0_rewards: List[float] = field(default_factory=list)
    agent_1_rewards: List[float] = field(default_factory=list)

    def push(
        self,
        agent_0_obs: np.ndarray,
        agent_1_obs: np.ndarray,
        agent_0_action: int,
        agent_1_action: int,
        agent_0_reward: float,
        agent_1_reward: float,
    ):
        self.agent_0_obs.append(agent_0_obs)
        self.agent_1_obs.append(agent_1_obs)
        self.agent_0_actions.append(agent_0_action)
        self.agent_1_actions.append(agent_1_action)
        self.agent_0_rewards.append(agent_0_reward)
        self.agent_1_rewards.append(agent_1_reward)

        if len(self.agent_0_obs) > self.K:
            self.loaded = True