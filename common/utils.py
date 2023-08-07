import os
import random

import gym_gridverse
import gymnasium as gym
import gymnasium_pomdps
import numpy as np
import torch
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment, GymStateWrapper
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)


def make_gridverse_env(env_id, seed, max_episode_len=None, mdp=False):
    """
    Generate environment with seeding/wrapping.

    Args:
        env_id: ID of the environment to use.
        seed: Seed to set for environment.
        max_episode_len: Episode timeout length.

    Returns:
        Generated environment.

    """
    inner_env = factory_env_from_yaml(env_id)
    state_representation = make_state_representation(
        "compact",
        inner_env.state_space,
    )
    observation_representation = make_observation_representation(
        "compact",
        inner_env.observation_space,
    )
    outer_env = OuterEnv(
        inner_env,
        state_representation=state_representation,
        observation_representation=observation_representation,
    )
    if mdp:
        env = gym.make(
            "GymV21Environment-v0", env=GymStateWrapper(GymEnvironment(outer_env))
        )
    else:
        env = gym.make("GymV21Environment-v0", env=GymEnvironment(outer_env))

    if max_episode_len is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def make_env(env_id, seed, max_episode_len=None):
    """
    Generate environment with seeding/wrapping.

    Args:
        env_id: ID of the environment to use.
        seed: Seed to set for environment.
        max_episode_len: Episode timeout length.

    Returns:
        Generated environment.

    """
    env = gym.make(env_id)
    if max_episode_len is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
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
    """
    Saves a checkpoint.

    Args:
        run_id: Wandb ID of run.
        checkpoint_dir: Directory to store checkpoint in.
        global_step: Timestep of training.
        models: State dict of models.
        optimizers: State dict of optimizers.
        replay_buffer: Replay buffer.
        rng_states: RNG states.
    """
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
    """
    Sets seeding for experiment.

    Args:
        seed: Seed.
        device: Device being used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
