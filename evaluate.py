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
# TEAMMATE_TYPE_DIRS = {
#     "POT": "place_in_pot",
#     "PLATE": "plate_pickup",
#     "SERVE": "soup_pickup",
# }


# TODO: plot histrogram of features for each teammate type, try to figure out what behaviors they exhibit
# TODO: compute classification accuracy on X, y with linear regression just to see how accurate that could be


def get_rollout(config, checkpoint_dir, team_dir_0, team_dir_1="place_in_pot", seed=0):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    agent_0 = BaseAgent(config, env, checkpoint_dir, team_dir_0, agent_idx=0)

    if config["STRATEGY"] == "base":
        agent_1 = BaseAgent(config, env, checkpoint_dir, team_dir_1, agent_idx=1)
    elif config["STRATEGY"] == "collab":
        agent_1 = CoLLABAgent(config, env, checkpoint_dir, TEAMMATE_TYPE_DIRS, agent_idx=1)
    elif config["STRATEGY"] == "recollab":
        agent_1 = ReCoLLABAgent(config, env, checkpoint_dir, TEAMMATE_TYPE_DIRS, agent_idx=1)
    elif config["STRATEGY"] == "plastic":
        agent_1 = PlasticAgent(config, env, checkpoint_dir, TEAMMATE_TYPE_DIRS, agent_idx=1)

    key = jax.random.PRNGKey(seed)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    key = jax.random.PRNGKey(0)
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

    pbar = tqdm(total=config["ENV_KWARGS"]["max_steps"])

    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        pi_0 = agent_0.act(obs, actions, rewards)
        pi_1 = agent_1.act(obs, actions, rewards)

        if config["STRATEGY"] in ["collab", "recollab"]:
            pred_teammates.append(agent_1.pred_teammate_type)
            true_teammate_type = [k for k, v in TEAMMATE_TYPE_DIRS.items() if v == team_dir_0][0]
            true_teammates.append(true_teammate_type)

        actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]
        rewards["agent_0"] = info["shaped_reward"]['agent_0']
        rewards["agent_1"] = info["shaped_reward"]['agent_1']
        all_rewards.append(reward['agent_1'])
        all_shaped_rewards.append(info["shaped_reward"]['agent_1'])
        t += 1
        pbar.update(1)

        actions = {k: ACTIONS[v] for k, v in actions.items()}

        state_seq.append(state)

    returns = sum(all_rewards)
    shaped_returns = sum(all_shaped_rewards)

    # Calculate teammate prediction accuracy
    if true_teammates:
        accuracy = sum(1 for pred, true in zip(pred_teammates, true_teammates) if pred == true) / len(true_teammates)
    else:
        accuracy = None

    print("Teammate Prediction Accuracy:", accuracy)

    return state_seq, returns, shaped_returns, accuracy


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
def main(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["LAYOUT_NAME"] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    checkpoint_dir = os.path.join(config['SAVE_PATH'], f'ippo_ff_overcooked_{layout_name}')

    team_names = ["no_shaping", "place_in_pot", "plate_pickup", "soup_pickup", "base"]
    # team_names = ["place_in_pot", "plate_pickup", "soup_pickup"]

    all_optimal_returns = []
    all_optimal_shaped_returns = []
    all_optimal_accuracies = []

    for team_name in team_names:
        state_seq, returns, shaped_returns, accuracy = get_rollout(
            config, checkpoint_dir, team_name, "no_shaping", seed=0
        )
        all_optimal_returns.append(returns)
        all_optimal_shaped_returns.append(shaped_returns)
        if accuracy is not None:
            all_optimal_accuracies.append(accuracy)

    print(f"Average optimal returns: {np.mean(all_optimal_returns)}")
    print(f"Average optimal shaped returns: {np.mean(all_optimal_shaped_returns)}")

    if all_optimal_accuracies:
        print("Optimal Teammate Prediction Accuracies: ", all_optimal_accuracies)
        print(f"Average optimal accuracy: {np.mean(all_optimal_accuracies)}")


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
    embed_documents(
        documents=all_documents,
        collection_name=config["DATABASE_NAME"],
        database_directory=config["DATABASE_DIR"],
        batch_size=10
    )


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked")
def analyze_features(config):
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    checkpoint_dir = os.path.join(config['SAVE_PATH'], f'ippo_ff_overcooked_{layout_name}')

    team_names = ["no_shaping", "place_in_pot", "plate_pickup", "soup_pickup", "base"]
    # team_names = ["place_in_pot", "plate_pickup", "soup_pickup"]
    target_names = ["Default","Pot","Plate","Serve","Mixed"]
    # target_names = ["Pot","Plate","Serve"]

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

    for teammate_type_idx, stats in summary.items():
        print(teammate_type_idx)
        teammate_type = target_names[int(teammate_type_idx)]
        print(f"Teammate type: {teammate_type}")
        pprint(stats)

    # pprint(summary)


if __name__ == "__main__":
    # analyze_features()
    # create_database()
    main()

    # Average returns given optimal team policies
    #   returns: 200
    #   shaped_returns: 73
    #
    # Average returns given sub-optimal no_shaping policy against multiple teammates
    #   returns: 52
    #   shaped_returns: 14
    #
    # Average returns given random
    #   returns: 44.0
    #   shaped_returns: 12.6
    #   accuracy: 0.22
    #
    # Average returns given CoLLAB
    #   returns: 96.0
    #   shaped_returns: 23.0
    #   accuracy: 0.58
    #
    # Average returns given ReCoLLAB
    #   returns: 128.0
    #   shaped_returns: 23.0
    #   accuracy: 0.77
    #
    # Average returns given PLASTIC
    #   returns: 
    #   shaped_returns: 
    #   accuracy: 

    # NOTE: The following results correspond to restricting teammate types to [POT, PLATE, SERVE]
    # Average returns given optimal team policies
    #   returns: 193.33
    #   shaped_returns: 56.67
    #   accuracy: N/A
    #
    # Average returns given sub-optimal no_shaping policy against multiple teammates
    #   returns: 80.0
    #   shaped_returns: 20.0
    #   accuracy: N/A
    #
    # Average returns given CoLLAB
    #   returns: 133.33
    #   shaped_returns: 18.33
    #   accuracy: 0.967