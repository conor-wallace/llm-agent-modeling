import numpy as np


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


def describe_observation(obs: np.ndarray) -> str:
    self_position = obs[0:2]
    self_velocity = obs[2:4]
    landmark_1_relative_position = obs[4:6]
    landmark_2_relative_position = obs[6:8]
    landmark_3_relative_position = obs[8:10]
    other_agent_1_relative_position = obs[10:12]
    other_agent_2_relative_position = obs[12:14]

    description = (
        f"Self Position: {self_position}, "
        f"Self Velocity: {self_velocity}, "
        f"Landmark 1 Relative Position: {landmark_1_relative_position}, "
        f"Landmark 2 Relative Position: {landmark_2_relative_position}, "
        f"Landmark 3 Relative Position: {landmark_3_relative_position}, "
        f"Other Agent 1 Relative Position: {other_agent_1_relative_position}, "
        f"Other Agent 2 Relative Position: {other_agent_2_relative_position}"
    )
    return description


def save_trajectory(path: str, trajectory: list):
    """Save the trajectory of env descriptions to a file.

    Args:
        path (str): The file path where the trajectory will be saved.
        trajectory (list): A list of strings representing the env trajectory.
    """
    with open(path, "w") as f:
        for obs in trajectory:
            f.write(obs + "\n")
    print(f"Trajectory saved to {path}")