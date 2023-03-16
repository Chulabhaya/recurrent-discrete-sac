from gymnasium.envs.registration import register
import gymnasium as gym

# Notation:
# F: full observed (original env)
# P: position/angle observed
# V: velocity observed

register(
    "Pendulum-F-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(
        env=gym.make("Pendulum-v1"), partially_obs_dims=[0, 1, 2]
    ),  # angle & velocity
    max_episode_steps=200,
)

register(
    "Pendulum-P-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("Pendulum-v1"), partially_obs_dims=[0, 1]),  # angle
    max_episode_steps=200,
)

register(
    "Pendulum-V-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("Pendulum-v1"), partially_obs_dims=[2]),  # velocity
    max_episode_steps=200,
)

register(
    "CartPole-F-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(
        env=gym.make("CartPole-v0"), partially_obs_dims=[0, 1, 2, 3]
    ),  # angle & velocity
    max_episode_steps=200,  # reward threshold for solving the task: 195
)

register(
    "CartPole-P-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("CartPole-v0"), partially_obs_dims=[0, 2]),
    max_episode_steps=200,
)

register(
    "CartPole-V-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("CartPole-v0"), partially_obs_dims=[1, 3]),
    max_episode_steps=200,
)

register(
    "LunarLander-F-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(
        env=gym.make("LunarLander-v2"), partially_obs_dims=list(range(8))
    ),  # angle & velocity
    max_episode_steps=1000,  # reward threshold for solving the task: 200
)

register(
    "LunarLander-P-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("LunarLander-v2"), partially_obs_dims=[0, 1, 4, 6, 7]),
    max_episode_steps=1000,
)

register(
    "LunarLander-V-v0",
    entry_point="pomdps.wrappers:POMDPWrapper",
    kwargs=dict(env=gym.make("LunarLander-v2"), partially_obs_dims=[2, 3, 5, 6, 7]),
    max_episode_steps=1000,
)
