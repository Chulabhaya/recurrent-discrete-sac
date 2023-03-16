import gymnasium as gym
from gymnasium import spaces
import numpy as np


class POMDPWrapper(gym.Wrapper):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, env, partially_obs_dims: list):
        super().__init__(env)
        self.partially_obs_dims = partially_obs_dims
        # can equal to the fully-observed env
        assert 0 < len(self.partially_obs_dims) <= self.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=self.observation_space.low[self.partially_obs_dims],
            high=self.observation_space.high[self.partially_obs_dims],
            dtype=np.float32,
        )

        if self.env.action_space.__class__.__name__ == "Box":
            self.act_continuous = True
            # if continuous actions, make sure in [-1, 1]
            # NOTE: policy won't use action_space.low/high, just set [-1,1]
            # this is a bad practice...
        else:
            self.act_continuous = False

    def get_obs(self, state):
        return state[self.partially_obs_dims].copy()

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)  # no kwargs
        return self.get_obs(state), info

    def step(self, action):
        if self.act_continuous:
            # recover the action
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        state, reward, terminated, truncated, info = self.env.step(action)

        return self.get_obs(state), reward, terminated, truncated, info
