import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from gymnasium import spaces
from collections import deque


class ReplayBuffer:
    """Replay buffer that stores timesteps of data for use with non-history-based
    algorithms."""

    def __init__(
        self,
        buffer_size,
        obs_space,
        action_space,
        device="cpu",
    ):
        """Initialize the episodic replay buffer.

        Parameters
        ----------
        buffer_size : int
            Maximum potential size of buffer in timesteps.
        obs_space : Gymnasium Discrete space.
        action_space : Gymnasium Discrete space.
        device : string
            Device on which samples from buffer should be returned.
        """
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
        """Saves content of buffer to allow for later reloading.

        Returns
        -------
        buffer_data : dict
            Dictionary containing current status of buffer.
        """
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
        """Load data from prior saved replay buffer.

        Parameters
        ----------
        buffer_data : dict
            Dictionary containing saved replay buffer data.
        """
        self.obs = buffer_data["obs"]
        self.actions = buffer_data["actions"]
        self.next_obs = buffer_data["next_obs"]
        self.rewards = buffer_data["rewards"]
        self.terminateds = buffer_data["terminateds"]
        self.truncateds = buffer_data["truncateds"]
        self.pos = buffer_data["pos"]
        self.full = buffer_data["full"]

    def add(self, obs, action, next_obs, reward, terminated, truncated):
        """Adds a timestep of data to replay buffer.

        Parameters
        ----------
        obs : int
            Observation.
        action : int
            Action.
        next_obs : int
            Next observation.
        reward : float
            Reward.
        terminated : bool
            Terminated status.
        truncated : bool
            Truncated status.
        """
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
        """Returns a batch of data from buffer.

        Parameters
        ----------
        batch_size : int
            Size of batch to sample from buffer.

        Returns
        -------
        batch_obs : tensor
            Batched observations.
        batch_actions : tensor
            Batched actions.
        batch_next_obs : tensor
            Batched observations.
        batch_rewards : tensor
            Batched rewards.
        batch_terminateds : tensor
            Batched terminations.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        return (
            torch.as_tensor(self.obs[batch_inds, :]).to(self.device),
            torch.as_tensor(self.actions[batch_inds, :]).to(self.device),
            torch.as_tensor(self.next_obs[batch_inds, :]).to(self.device),
            torch.as_tensor(self.rewards[batch_inds]).unsqueeze(1).to(self.device),
            torch.as_tensor(self.terminateds[batch_inds]).unsqueeze(1).to(self.device),
        )


class EpisodicReplayBuffer:
    """Replay buffer that stores complete episodes for use with history-based
    learning algorithms."""

    def __init__(
        self,
        buffer_size,
        device="cpu",
    ):
        """Initialize the episodic replay buffer.

        Parameters
        ----------
        buffer_size : int
            Maximum potential size of buffer in timesteps.
        device : string
            Device on which samples from buffer should be returned.
        """
        self.buffer_size = buffer_size

        # Store a list of episodes
        self.obs = deque()
        self.actions = deque()
        self.next_obs = deque()
        self.rewards = deque()
        self.terminateds = deque()
        self.truncateds = deque()

        # Track on-going episode before adding it to
        # overall list
        self.ongoing_obs = deque()
        self.ongoing_actions = deque()
        self.ongoing_next_obs = deque()
        self.ongoing_rewards = deque()
        self.ongoing_terminateds = deque()
        self.ongoing_truncateds = deque()

        self.timesteps_in_buffer = 0
        self.device = device

    def save_buffer(self):
        """Saves content of buffer to allow for later reloading.

        Returns
        -------
        buffer_data : dict
            Dictionary containing current status of buffer.
        """
        buffer_data = {
            "obs": self.obs,
            "actions": self.actions,
            "next_obs": self.next_obs,
            "rewards": self.rewards,
            "terminateds": self.terminateds,
            "truncateds": self.truncateds,
            "timesteps_in_buffer": self.timesteps_in_buffer,
        }

        return buffer_data

    def load_buffer(self, buffer_data):
        """Load data from prior saved replay buffer.

        Parameters
        ----------
        buffer_data : dict
            Dictionary containing saved replay buffer data.
        """
        self.obs = buffer_data["obs"]
        self.actions = buffer_data["actions"]
        self.next_obs = buffer_data["next_obs"]
        self.rewards = buffer_data["rewards"]
        self.terminateds = buffer_data["terminateds"]
        self.truncateds = buffer_data["truncateds"]
        self.timesteps_in_buffer = buffer_data["timesteps_in_buffer"]

    def add(self, obs, action, next_obs, reward, terminated, truncated):
        """Adds a timestep of data to episodic replay buffer.

        Parameters
        ----------
        obs : int
            Observation.
        action : int
            Action.
        next_obs : int
            Next observation.
        reward : float
            Reward.
        terminated : bool
            Terminated status.
        truncated : bool
            Truncated status.
        """

        # Update on-going episode with new timestep
        self.ongoing_obs.append(obs)
        self.ongoing_actions.append(action)
        self.ongoing_next_obs.append(next_obs)
        self.ongoing_rewards.append(reward)
        self.ongoing_terminateds.append(terminated)
        self.ongoing_truncateds.append(truncated)

        # If episode is over, then we add to buffer
        if terminated or truncated:
            self.obs.append(self.ongoing_obs)
            self.actions.append(self.ongoing_actions)
            self.next_obs.append(self.ongoing_next_obs)
            self.rewards.append(self.ongoing_rewards)
            self.terminateds.append(self.ongoing_terminateds)
            self.truncateds.append(self.ongoing_truncateds)

            # Update buffer size tracking
            self.timesteps_in_buffer += len(self.ongoing_obs)

            # Reset on-going episode tracking
            self.ongoing_obs = deque()
            self.ongoing_actions = deque()
            self.ongoing_next_obs = deque()
            self.ongoing_rewards = deque()
            self.ongoing_terminateds = deque()
            self.ongoing_truncateds = deque()

        # Ensure that buffer size isn't over the limit
        # by removing the earliest episodes in the buffer
        while self.timesteps_in_buffer > self.buffer_size:
            removed_obs = self.obs.popleft()
            self.actions.popleft()
            self.next_obs.popleft()
            self.rewards.popleft()
            self.terminateds.popleft()
            self.truncateds.popleft()

            self.timesteps_in_buffer -= len(removed_obs)

    def sample(self, batch_size=256, history_length=None):
        """Sample batch of episodes from episodic replay buffer.

        Parameters
        ----------
        batch_size : int
            Size of batch to sample from buffer.
        history_length : int
            Sequence length to sample, if None then
            full episode will be sampled.

        Returns
        -------
        batch_obs : tensor
            Batched and padded episode observations.
        batch_actions : tensor
            Batched and padded episode actions.
        batch_next_obs : tensor
            Batched and padded episode next observations.
        batch_rewards : tensor
            Batched and padded episode rewards.
        batch_terminateds : tensor
            Batched and padded episode terminations.
        seq_lengths : tensor
            Sequence lengths to keep track of padding.
        """
        if history_length:
            # Generate indices for random episodes
            upper_bound = len(self.obs)
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)

            # Deques for storing batch to return
            batch_obs = deque()
            batch_actions = deque()
            batch_next_obs = deque()
            batch_rewards = deque()
            batch_terminateds = deque()

            # Generate batch
            for i in range(batch_size):
                # Generate random index in episode for sampling history
                # sequence
                episode_length = len(self.obs[batch_inds[i]])
                episode_ind_start = np.random.randint(0, episode_length)
                episode_ind_end = min(
                    episode_ind_start + history_length, episode_length
                )

                batch_obs.append(
                    torch.as_tensor(
                        np.array(self.obs[batch_inds[i]])[
                            episode_ind_start:episode_ind_end
                        ]
                    )
                )
                batch_actions.append(
                    torch.as_tensor(
                        np.array(self.actions[batch_inds[i]])[
                            episode_ind_start:episode_ind_end
                        ]
                    )
                )
                batch_next_obs.append(
                    torch.as_tensor(
                        np.array(self.next_obs[batch_inds[i]])[
                            episode_ind_start:episode_ind_end
                        ]
                    )
                )
                batch_rewards.append(
                    torch.as_tensor(
                        np.array(self.rewards[batch_inds[i]], dtype=np.float32)[
                            episode_ind_start:episode_ind_end
                        ]
                    )
                )
                batch_terminateds.append(
                    torch.as_tensor(
                        np.array(self.terminateds[batch_inds[i]])[
                            episode_ind_start:episode_ind_end
                        ]
                    )
                )

        else:
            # Generate indices for random episodes
            upper_bound = len(self.obs)
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)

            # Deques for storing batch to return
            batch_obs = deque()
            batch_actions = deque()
            batch_next_obs = deque()
            batch_rewards = deque()
            batch_terminateds = deque()

            # Generate batch
            for i in range(batch_size):
                batch_obs.append(torch.as_tensor(np.array(self.obs[batch_inds[i]])))
                batch_actions.append(
                    torch.as_tensor(np.array(self.actions[batch_inds[i]]))
                )
                batch_next_obs.append(
                    torch.as_tensor(np.array(self.next_obs[batch_inds[i]]))
                )
                batch_rewards.append(
                    torch.as_tensor(
                        np.array(self.rewards[batch_inds[i]], dtype=np.float32)
                    )
                )
                batch_terminateds.append(
                    torch.as_tensor(np.array(self.terminateds[batch_inds[i]]))
                )

        # Create padded arrays of history
        seq_lengths = torch.LongTensor(list(map(len, batch_obs)))
        batch_obs = pad_sequence(batch_obs).to(self.device)
        batch_actions = torch.unsqueeze(pad_sequence(batch_actions).to(self.device), 2)
        batch_next_obs = pad_sequence(batch_next_obs).to(self.device)
        batch_rewards = torch.unsqueeze(pad_sequence(batch_rewards).to(self.device), 2)
        batch_terminateds = torch.unsqueeze(
            pad_sequence(batch_terminateds).to(self.device), 2
        ).long()

        return (
            batch_obs,
            batch_actions,
            batch_next_obs,
            batch_rewards,
            batch_terminateds,
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
