import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        obs_space,
        action_space,
        device="cpu",
        handle_timeout_termination=True,
    ):
        self.buffer_size = buffer_size

        self.obs_space = obs_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(obs_space)
        self.action_dim = get_action_dim(action_space)

        self.obs = np.zeros((self.buffer_size,) + self.obs_shape, dtype=obs_space.dtype)
        self.next_obs = np.zeros(
            (self.buffer_size,) + self.obs_shape, dtype=obs_space.dtype
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size), dtype=np.float32)

        self.pos = 0
        self.full = False
        self.device = device

    def save_buffer(self):
        buffer_data = {
            "obs": self.obs,
            "next_obs": self.next_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "handle_timeout_termination": self.handle_timeout_termination,
            "timeouts": self.timeouts,
            "pos": self.pos,
            "full": self.full,
        }

        return buffer_data

    def load_buffer(self, buffer_data):
        self.obs = buffer_data["obs"]
        self.next_obs = buffer_data["next_obs"]
        self.actions = buffer_data["actions"]
        self.rewards = buffer_data["rewards"]
        self.dones = buffer_data["dones"]
        self.handle_timeout_termination = buffer_data["handle_timeout_termination"]
        self.timeouts = buffer_data["timeouts"]
        self.pos = buffer_data["pos"]
        self.full = buffer_data["full"]

    def add(self, obs, next_obs, action, reward, done, info):
        # Copy to avoid modification by reference
        self.obs[self.pos] = np.array(obs).copy()
        self.next_obs[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(info.get("TimeLimit.truncated", False))

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
            torch.as_tensor(self.dones[batch_inds] * (1 - self.timeouts[batch_inds]))
            .unsqueeze(1)
            .to(self.device),
            torch.as_tensor(self.rewards[batch_inds]).unsqueeze(1).to(self.device),
        )

    def sample_history(self, batch_size, history_length=None):
        # Generate random sample indices from dataset
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Lists for storing histories
        obs_histories = []
        actions_histories = []
        next_obs_histories = []
        dones_histories = []
        rewards_histories = []

        # Get locations of all dones
        dones = np.argwhere(self.dones == 1)[:, 0]
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
                action_history = torch.as_tensor(
                    self.actions[start_timestep : end_timestep + 1]
                )
                next_obs_history = torch.as_tensor(
                    self.next_obs[start_timestep : end_timestep + 1]
                )
                dones_history = torch.as_tensor(
                    self.dones[start_timestep : end_timestep + 1]
                    * (1 - self.timeouts[start_timestep : end_timestep + 1])
                )
                rewards_history = torch.as_tensor(
                    self.rewards[start_timestep : end_timestep + 1]
                )

                # Append to proper lists
                obs_histories.append(obs_history)
                actions_histories.append(action_history)
                next_obs_histories.append(next_obs_history)
                dones_histories.append(dones_history)
                rewards_histories.append(rewards_history)
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
                action_history = torch.as_tensor(
                    self.actions[start_timestep : end_timestep + 1]
                )
                next_obs_history = torch.as_tensor(
                    self.next_obs[start_timestep : end_timestep + 1]
                )
                dones_history = torch.as_tensor(
                    self.dones[start_timestep : end_timestep + 1]
                    * (1 - self.timeouts[start_timestep : end_timestep + 1])
                )
                rewards_history = torch.as_tensor(
                    self.rewards[start_timestep : end_timestep + 1]
                )

                # Append to proper lists
                obs_histories.append(obs_history)
                actions_histories.append(action_history)
                next_obs_histories.append(next_obs_history)
                dones_histories.append(dones_history)
                rewards_histories.append(rewards_history)

        # Create padded arrays of history
        seq_lengths = torch.LongTensor(list(map(len, obs_histories)))
        obs_histories = pad_sequence(obs_histories).to(self.device)
        actions_histories = pad_sequence(actions_histories).to(self.device)
        next_obs_histories = pad_sequence(next_obs_histories).to(self.device)
        dones_histories = torch.unsqueeze(
            pad_sequence(dones_histories).to(self.device), 2
        )
        rewards_histories = torch.unsqueeze(
            pad_sequence(rewards_histories).to(self.device), 2
        )

        return (
            obs_histories,
            actions_histories,
            next_obs_histories,
            dones_histories,
            rewards_histories,
            seq_lengths,
        )
