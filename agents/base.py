import os
from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from jaxmarl.environments.overcooked import overcooked_layouts
from safetensors.flax import load_file

from .history import TeammateHistory


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class BaseAgent:
    def __init__(
        self, config, env, checkpoint_dir, team_dir, agent_idx
    ):
        self.config = config
        self.env = env
        self.checkpoint_dir = checkpoint_dir
        self.team_dir = team_dir
        self.agent_idx = agent_idx

        self.hist = TeammateHistory(K=self.config["K"])

        self.network = ActorCritic(
            self.env.action_space(f"agent_{self.agent_idx}").n,
            activation=self.config["ACTIVATION"]
        )
        self.load_params()

    def load_params(self):
        flat_params = load_file(os.path.join(self.checkpoint_dir, self.team_dir) + f'/model_{self.agent_idx}.safetensors')

        # Restructure parameters to the nested format expected by Flax
        self.network_params = {'params': {}}
        for key, value in flat_params.items():
            parts = key.split(',')
            if len(parts) == 3:  # expected format: 'params,Module,param'
                collection, module, param = parts
                if module not in self.network_params[collection]:
                    self.network_params[collection][module] = {}
                self.network_params[collection][module][param] = value.squeeze(0)

    def act(self, obs, prev_actions, prev_rewards):
        obs_0 = obs["agent_0"]
        obs_1 = obs["agent_1"]
        prev_action_0 = prev_actions["agent_0"]
        prev_action_1 = prev_actions["agent_1"]
        prev_reward_0 = prev_rewards["agent_0"]
        prev_reward_1 = prev_rewards["agent_1"]

        # Update teammate observations
        obs_desc_0 = None
        obs_desc_1 = None

        self.hist.push(
            obs_0,
            obs_1,
            prev_action_0,
            prev_action_1,
            prev_reward_0,
            prev_reward_1
        )

        obs_i = obs[f"agent_{self.agent_idx}"].flatten()
        pi, _ = self.network.apply(self.network_params, obs_i)
        return pi