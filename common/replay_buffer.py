import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from gymnasium import spaces


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        obs_space,
        action_space,
        device="cpu",
    ):
        self.buffer_size = buffer_size

        self.obs_space = obs_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(obs_space)
        self.action_dim = get_action_dim(action_space)

        self.obs = np.zeros((self.buffer_size,) + self.obs_shape, dtype=obs_space.dtype)
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=action_space.dtype
        )
        self.next_obs = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=obs_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.terminateds = np.zeros((self.buffer_size), dtype=np.float32)
        self.truncateds = np.zeros((self.buffer_size), dtype=np.float32)

        self.pos = 0
        self.full = False
        self.device = device

    def save_buffer(self):
        buffer_data = {
            "obs": self.obs,
            "actions": self.actions,
            "next_obs": self.next_obs,
            "rewards": self.rewards,
            "terminateds": self.terminateds,
            "truncateds": self.truncateds,
            "pos": self.pos,
            "full": self.full,
        }

        return buffer_data

    def load_buffer(self, buffer_data):
        self.obs = buffer_data["obs"]
        self.actions = buffer_data["actions"]
        self.next_obs = buffer_data["next_obs"]
        self.rewards = buffer_data["rewards"]
        self.terminateds = buffer_data["terminateds"]
        self.truncateds = buffer_data["truncateds"]
        self.pos = buffer_data["pos"]
        self.full = buffer_data["full"]

    def add(self, obs, action, next_obs, reward, terminated, truncated):
        # Copy to avoid modification by reference
        self.obs[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.next_obs[self.pos] = np.array(next_obs).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.terminateds[self.pos] = np.array(terminated).copy()
        self.truncateds[self.pos] = np.array(truncated).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        return (
            torch.as_tensor(self.obs[batch_inds, :]).to(self.device),
            torch.as_tensor(self.actions[batch_inds, :]).to(self.device),
            torch.as_tensor(self.next_obs[batch_inds, :]).to(self.device),
            torch.as_tensor(self.rewards[batch_inds]).unsqueeze(1).to(self.device),
            torch.as_tensor(self.terminateds[batch_inds]).unsqueeze(1).to(self.device),
        )

    def sample_history(self, batch_size, history_length=None):
        # Generate random sample indices from dataset
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Lists for storing histories
        obs_histories = []
        actions_histories = []
        next_obs_histories = []
        rewards_histories = []
        terminateds_histories = []

        # Get locations of all dones (both terminated and truncated)
        dones = self.terminateds + self.truncateds
        dones = np.argwhere(dones == 1)[:, 0]
        # Generate batch of histories
        for i in range(batch_size):
            # Get index
            end_timestep = batch_inds[i]
            if history_length is None:
                # Get closest done to index that is less than
                # the index value
                if end_timestep <= dones[0]:
                    start_timestep = 0
                else:
                    previous_done_timestep = dones[dones < end_timestep].max()
                    start_timestep = previous_done_timestep + 1

                # Get full trajectory up to index timestep from start of the episode
                obs_history = torch.as_tensor(
                    self.obs[start_timestep : end_timestep + 1]
                )
                actions_history = torch.as_tensor(
                    self.actions[start_timestep : end_timestep + 1]
                )
                next_obs_history = torch.as_tensor(
                    self.next_obs[start_timestep : end_timestep + 1]
                )
                rewards_history = torch.as_tensor(
                    self.rewards[start_timestep : end_timestep + 1]
                )
                terminateds_history = torch.as_tensor(
                    self.terminateds[start_timestep : end_timestep + 1]
                )

                # Append to proper lists
                obs_histories.append(obs_history)
                actions_histories.append(actions_history)
                next_obs_histories.append(next_obs_history)
                rewards_histories.append(rewards_history)
                terminateds_histories.append(terminateds_history)
            else:
                # Go backwards for history length to get trajectory history
                # If going backwards a history length would go beyond
                # previous done, then stop history early
                if end_timestep <= dones[0]:
                    if end_timestep + 1 - history_length <= 0:
                        start_timestep = 0
                    else:
                        start_timestep = end_timestep + 1 - history_length
                else:
                    previous_done_timestep = dones[dones < end_timestep].max()
                    # Calculate start timestep for history
                    # The +1 is to account for array end indexing
                    start_timestep = end_timestep + 1 - history_length
                    if previous_done_timestep >= start_timestep:
                        start_timestep = previous_done_timestep + 1

                # Get specific history
                obs_history = torch.as_tensor(
                    self.obs[start_timestep : end_timestep + 1]
                )
                actions_history = torch.as_tensor(
                    self.actions[start_timestep : end_timestep + 1]
                )
                next_obs_history = torch.as_tensor(
                    self.next_obs[start_timestep : end_timestep + 1]
                )
                rewards_history = torch.as_tensor(
                    self.rewards[start_timestep : end_timestep + 1]
                )
                terminateds_history = torch.as_tensor(
                    self.terminateds[start_timestep : end_timestep + 1]
                )

                # Append to proper lists
                obs_histories.append(obs_history)
                actions_histories.append(actions_history)
                next_obs_histories.append(next_obs_history)
                rewards_histories.append(rewards_history)
                terminateds_histories.append(terminateds_history)

        # Create padded arrays of history
        seq_lengths = torch.LongTensor(list(map(len, obs_histories)))
        obs_histories = pad_sequence(obs_histories).to(self.device)
        actions_histories = pad_sequence(actions_histories).to(self.device)
        next_obs_histories = pad_sequence(next_obs_histories).to(self.device)
        terminateds_histories = torch.unsqueeze(
            pad_sequence(terminateds_histories).to(self.device), 2
        )
        rewards_histories = torch.unsqueeze(
            pad_sequence(rewards_histories).to(self.device), 2
        )

        return (
            obs_histories,
            actions_histories,
            next_obs_histories,
            rewards_histories,
            terminateds_histories,
            seq_lengths,
        )


def get_obs_shape(observation_space):
    """
    Get the shape of the observation (useful for the buffers).
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_action_dim(action_space):
    """
    Get the dimension of the action space.
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
