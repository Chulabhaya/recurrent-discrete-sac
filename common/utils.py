import os
import random

import gymnasium as gym
import gymnasium_pomdps
import numpy as np
import torch
from gymnasium_pomdps.wrappers.mdp import MDP
from minigrid.wrappers import FullyObsWrapper

import simple_pomdps
from minigrid_action_wrappers.movement_action_mask import MovementActionMask


def make_env(env_id, seed, capture_video, run_name, max_episode_len=None):
    """Generates seeded environment.

    Parameters
    ----------
    env_id : string
        Name of Gym environment.
    seed : int
        Seed.
    idx : int
        Whether to record videos or not.
    capture_video : boolean
        Whether to record videos or not.
    run_name : string
        Name of run to be used for video.

    Returns
    -------
    env : gym environment
        Gym environment to be used for learning.
    """
    env = MovementActionMask(FullyObsWrapper(gym.make(env_id, max_steps=100)))
    if max_episode_len is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def save(
    run_id,
    checkpoint_dir,
    global_step,
    models,
    optimizers,
    replay_buffer,
    rng_states,
):
    save_dir = checkpoint_dir + run_id + "/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # Prevent permission issues when writing to this directory
        # after resuming a training job
        os.chmod(save_dir, 0o777)

    save_path = save_dir + "global_step_" + str(global_step) + ".pth"
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": models,
            "optimizer_state_dict": optimizers,
            "replay_buffer": replay_buffer,
            "rng_states": rng_states,
        },
        save_path,
    )


def set_seed(seed, device):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
