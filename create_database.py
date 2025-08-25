import os
from pprint import pprint
from typing import Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jaxmarl.environments.overcooked import overcooked_layouts
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from safetensors.flax import load_file
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from agents import BaseAgent, CoLLABAgent, ReCoLLABAgent, PlasticAgent
from agents.database import create_document, embed_documents
from agents.features import fingerprint_from_window, summarize_results


ACTIONS = ["UP","DOWN","LEFT","RIGHT","STAY","INTERACT"]
TEAMMATE_TYPE_DIRS = {
    "DEFAULT": "no_shaping",
    "POT": "place_in_pot",
    "PLATE": "plate_pickup",
    "SERVE": "soup_pickup",
    "MIXED": "base"
}


def get_rollout_features(config, checkpoint_dir, team_dir_0, team_dir_1="base", stop_after_K=100, seed=0):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    agent_0 = BaseAgent(config, env, checkpoint_dir, team_dir_0, agent_idx=0)
    agent_1 = BaseAgent(config, env, checkpoint_dir, team_dir_1, agent_idx=1)

    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    key = jax.random.PRNGKey(seed)
    key, key_r, key_a = jax.random.split(key, 3)

    obs, state = env.reset(key_r)
    state_seq = [state]
    actions = {"agent_0": None, "agent_1": None}
    rewards = {"agent_0": 0.0, "agent_1": 0.0}
    all_rewards = []
    all_shaped_rewards = []
    pred_teammates = []
    true_teammates = []
    t = 0

    while t < stop_after_K:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        pi_0 = agent_0.act(obs, actions, rewards)
        pi_1 = agent_1.act(obs, actions, rewards)

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]
        rewards = {"agent_0": float(info["shaped_reward"]['agent_0']), "agent_1": float(info["shaped_reward"]['agent_1'])}
        all_rewards.append(reward['agent_1'])
        all_shaped_rewards.append(info["shaped_reward"]['agent_1'])
        t += 1

        actions = {k: ACTIONS[v] for k, v in actions.items()}

    return agent_0.hist, agent_1.hist


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked")
def create_database(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    checkpoint_dir = os.path.join(config['SAVE_PATH'], f'ippo_ff_overcooked_{layout_name}')

    team_names = ["no_shaping", "place_in_pot", "plate_pickup", "soup_pickup", "base"]

    all_documents = []
    pbar = tqdm(total=config["NUM_SEEDS"] * len(team_names))
    for team_name in team_names:
        teammate_type = next(k for k, v in TEAMMATE_TYPE_DIRS.items() if v == team_name)
        for j in range(config["NUM_SEEDS"]):
            agent_0_hist, agent_1_hist = get_rollout_features(
                config=config,
                checkpoint_dir=checkpoint_dir,
                team_dir_0=team_name,
                team_dir_1="no_shaping",
                stop_after_K=20,
                seed=j
            )
            agent_features_dict_0 = fingerprint_from_window(
                agent_0_hist.agent_0_obs,
                agent_0_hist.agent_0_actions,
                agent_0_hist.agent_0_rewards
            )

            document = create_document(agent_features_dict_0, teammate_type)
            all_documents.append(document)
            pbar.update(1)

    print(f"Number of documents = {len(all_documents)}")

    database_dir = os.path.join(config['DATABASE_PATH'], f'ippo_ff_overcooked_{layout_name}')
    database_name = f'ippo_ff_overcooked_{layout_name}'

    embed_documents(
        documents=all_documents,
        collection_name=database_name,
        database_directory=database_dir,
        batch_size=10
    )


if __name__ == "__main__":
    create_database()