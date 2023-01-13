import gym
from gym_pomdps.wrappers.resetobservation import ResetObservationWrapper


def make_env(env_id, seed, idx, capture_video, run_name):
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

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_env_gym_pomdp(env_id, seed, idx, capture_video, run_name, max_episode_len):
    """Generates seeded environment for discrete Gym POMDPs that need
    additional env wrappers.

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

    def thunk():
        env = gym.wrappers.TimeLimit(
            ResetObservationWrapper(gym.make(env_id)), max_episode_steps=max_episode_len
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
