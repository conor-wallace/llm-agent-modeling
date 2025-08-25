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
def analyze_features(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    checkpoint_dir = os.path.join(config['SAVE_PATH'], f'ippo_ff_overcooked_{layout_name}')

    if config["USE_ALL_TEAMMATES"]:
        team_names = ["no_shaping", "place_in_pot", "plate_pickup", "soup_pickup", "base"]
        target_names = ["Default","Pot","Plate","Serve","Mixed"]
    else:
        team_names = ["place_in_pot", "plate_pickup", "soup_pickup"]
        target_names = ["Pot","Plate","Serve"]

    X = []
    y = []

    pbar = tqdm(total=config["NUM_SEEDS"] * len(team_names))
    for i, team_name in enumerate(team_names):
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

            X.append(list(agent_features_dict_0.values()))
            y.append(i)

            pbar.update(1)

    X = np.array(X)
    y = np.array(y)

    # X: N x F matrix of your fingerprints (first K steps); y: type indices 0..4
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    feat_names = [
        "cumulative_reward",
        "first_destination",
        "near_pot_steps",
        "near_plate_pile_steps",
        "near_onion_pile_steps",
        "near_window_steps",
        "interacts_with_pot",
        "interacts_with_plate_pile",
        "interacts_with_onion_pile",
        "interacts_at_ready_pot_or_soup",
        "unique_tasks_attempted_in_first_K",
        "time_to_first_interact_pot",
        "time_to_first_interact_plate",
        "time_to_first_interact_onion",
        "time_to_first_interact_serve",
        "dwell_pot",
        "dwell_plate",
        "dwell_onion",
        "dwell_window",
        "blocked_events",
        "handoff_events"
    ]

    rank = sorted(zip(mi, feat_names), reverse=True)
    print("MI ranking:")
    for v,n in rank: print(f"{n:>18s}: {v:0.3f}")

    clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0))
    ]).fit(X, y)

    print(export_text(clf.named_steps["tree"], feature_names=feat_names))

    # Fit multinomial logistic regression
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf.fit(X, y)

    # Predictions
    y_pred = clf.predict(X)

    # Overall accuracy
    acc = accuracy_score(y, y_pred)
    print("Overall accuracy:", acc)

    # Accuracy per teammate type
    print(classification_report(y, y_pred, target_names=target_names))

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.show()
    plt.savefig("confusion_matrix.png")

    summary = summarize_results(X, y, feat_names + ["teammate_type"])

    behavior_rubric = ""
    for teammate_type_idx, stats in summary.items():
        teammate_type = target_names[int(teammate_type_idx)].upper()
        print(f"Teammate type: {teammate_type}")
        pprint(stats)

        headers = ["cumulative_reward", "dwell_onion", "dwell_plate", "dwell_pot", "dwell_window", "near_onion_pile_steps", "near_plate_pile_steps", "near_pot_steps", "near_window_steps"]
        values = [
            stats["cumulative_reward_mean±std"],
            stats["dwell_onion_mean±std"],
            stats["dwell_plate_mean±std"],
            stats["dwell_pot_mean±std"],
            stats["dwell_window_mean±std"],
            stats["near_onion_pile_steps_mean±std"],
            stats["near_plate_pile_steps_mean±std"],
            stats["near_pot_steps_mean±std"],
            stats["near_window_steps_mean±std"]
        ]

        behavior_rubric += f"{teammate_type}\n"
        behavior_rubric += "  ".join(f"{header:<21}" for header in headers) + "\n"
        behavior_rubric += "  ".join(f"{str(value):<21}" for value in values) + "\n\n"

    if config["USE_ALL_TEAMMATES"]:
        rubric_path = os.path.join(config['RUBRIC_PATH'], f'ippo_ff_overcooked_{layout_name}', 'behavior_rubric.txt')
    else:
        rubric_path = os.path.join(config['RUBRIC_PATH'], f'ippo_ff_overcooked_{layout_name}', 'behavior_rubric_subset.txt')

    os.makedirs(os.path.dirname(rubric_path), exist_ok=True)

    with open(rubric_path, 'w') as f:
        f.write(behavior_rubric)


if __name__ == "__main__":
    analyze_features()