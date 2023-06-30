import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from collections import deque


class ReplayBuffer:
    """Replay buffer that stores complete episodes for use with history-based
    learning algorithms."""

    def __init__(
        self,
        size,
        episodic=False,
        stateful=False,
        device="cpu",
    ):
        """
        Initialize replay buffer.

        Args:
            buffer_size: Size of the buffer.
            episodic: Whether the buffer should be an episodic buffer.
            stateful: Whether the buffer should store states in addition
                to observations.
            device: What device to return buffer samples on.
        """
        # Set size of buffer
        self._buffer_size = size

        # Set type of buffer
        self._episodic = episodic
        self._stateful = stateful

        # Storage for data
        self._obs = deque()
        self._actions = deque()
        self._next_obs = deque()
        self._rewards = deque()
        self._terminateds = deque()
        self._truncateds = deque()
        # Store extra state data if stateful buffer
        if self._stateful:
            self._states = deque()
            self._next_states = deque()

        # If episodic buffer, then need intermediate storage
        if self._episodic:
            # Track on-going episode before adding it to
            # overall storage
            self._ongoing_obs = deque()
            self._ongoing_actions = deque()
            self._ongoing_next_obs = deque()
            self._ongoing_rewards = deque()
            self._ongoing_terminateds = deque()
            self._ongoing_truncateds = deque()
            # Store extra state data if stateful buffer
            if self._stateful:
                self._ongoing_states = deque()
                self._ongoing_next_states = deque()

        # Track number of timesteps stored in buffer
        self._timesteps_in_buffer = 0

        # Set device on which to return samples from buffer
        self._device = device

    def save_buffer(self):
        """
        Save contents of buffer.

        Returns:
            Dictionary containing all buffer data.
        """
        buffer_data = {
            "obs": self._obs,
            "actions": self._actions,
            "next_obs": self._next_obs,
            "rewards": self._rewards,
            "terminateds": self._terminateds,
            "truncateds": self._truncateds,
            "timesteps_in_buffer": self._timesteps_in_buffer,
        }
        # Store extra state data if stateful buffer
        if self._stateful:
            buffer_data["states"] = self._states
            buffer_data["next_states"] = self._next_states

        return buffer_data

    def load_buffer(self, buffer_data):
        """
        Load data into buffer.

        Args:
            buffer_data: Data to load into buffer.
        """
        self._obs = buffer_data["obs"]
        self._actions = buffer_data["actions"]
        self._next_obs = buffer_data["next_obs"]
        self._rewards = buffer_data["rewards"]
        self._terminateds = buffer_data["terminateds"]
        self._truncateds = buffer_data["truncateds"]
        self._timesteps_in_buffer = buffer_data["timesteps_in_buffer"]
        # Load extra state data if stateful buffer
        if self._stateful:
            self._states = buffer_data["states"]
            self._next_states = buffer_data["next_states"]

    def add(
        self,
        obs,
        action,
        next_obs,
        reward,
        terminated,
        truncated,
        state=None,
        next_state=None,
    ):
        """
        Add a timestep into buffer.

        Args:
            obs: Observation.
            action: Action.
            next_obs: Next observation.
            reward: Reward.
            terminated: Terminated status.
            truncated: Truncated status.
            state: State, optional.
            next_state: Next state, optional.
        """
        # If episodic buffer then use intermediary episode storage
        if self._episodic:
            # Update on-going episode with new timestep
            self._ongoing_obs.append(obs)
            self._ongoing_actions.append(action)
            self._ongoing_next_obs.append(next_obs)
            self._ongoing_rewards.append(reward)
            self._ongoing_terminateds.append(terminated)
            self._ongoing_truncateds.append(truncated)
            if self._stateful:
                self._ongoing_states.append(state)
                self._ongoing_next_states.append(next_state)

            # If episode is over, then we add to buffer
            if terminated or truncated:
                self._obs.append(self._ongoing_obs)
                self._actions.append(self._ongoing_actions)
                self._next_obs.append(self._ongoing_next_obs)
                self._rewards.append(self._ongoing_rewards)
                self._terminateds.append(self._ongoing_terminateds)
                self._truncateds.append(self._ongoing_truncateds)
                if self._stateful:
                    self._states.append(self._ongoing_states)
                    self._next_states.append(self._ongoing_next_states)

                # Update buffer size tracking
                self._timesteps_in_buffer += len(self._ongoing_obs)

                # Reset on-going episode tracking
                self._ongoing_obs = deque()
                self._ongoing_actions = deque()
                self._ongoing_next_obs = deque()
                self._ongoing_rewards = deque()
                self._ongoing_terminateds = deque()
                self._ongoing_truncateds = deque()
                if self._stateful:
                    self._ongoing_states = deque()
                    self._ongoing_next_states = deque()
        else:
            # Update storage with new samples
            self._obs.append(obs)
            self._actions.append(action)
            self._next_obs.append(next_obs)
            self._rewards.append(reward)
            self._terminateds.append(terminated)
            self._truncateds.append(truncated)
            if self._stateful:
                self._states.append(state)
                self._next_states.append(next_state)

            # Update buffer size tracking
            self._timesteps_in_buffer += 1

        # Ensure that buffer size isn't over the limit
        # by removing the earliest episodes/samples in the buffer
        while self._timesteps_in_buffer > self._buffer_size:
            removed_obs = self._obs.popleft()
            self._actions.popleft()
            self._next_obs.popleft()
            self._rewards.popleft()
            self._terminateds.popleft()
            self._truncateds.popleft()
            if self._stateful:
                self._states.popleft()
                self._next_states.popleft()

            if self._episodic:
                self._timesteps_in_buffer -= len(removed_obs)
            else:
                self._timesteps_in_buffer -= 1

    def sample(self, batch_size=256, history_length=None):
        """
        Sample batch of episodes/samples from replay buffer.

        Args:
            batch_size: Size of batch to sample from buffer.
            history_length: Sequence length to sample if using
                episodic buffer; if None then full episode will
                be sampled.

        Returns:
            Batch of tensors of samples/episodes.
        """
        # Generate indices for random samples/episodes
        upper_bound = len(self._obs)
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Deques for storing batch to return
        batch_obs = deque()
        batch_actions = deque()
        batch_next_obs = deque()
        batch_rewards = deque()
        batch_terminateds = deque()
        if self._stateful:
            batch_states = deque()
            batch_next_states = deque()

        # Generate episodic batch
        if self._episodic:
            if history_length:
                for i in range(batch_size):
                    # Generate random index in episode for sampling history
                    # sequence
                    episode_length = len(self._obs[batch_inds[i]])
                    episode_ind_start = np.random.randint(0, episode_length)
                    episode_ind_end = min(
                        episode_ind_start + history_length, episode_length
                    )

                    batch_obs.append(
                        torch.as_tensor(
                            np.array(self._obs[batch_inds[i]])[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )
                    batch_actions.append(
                        torch.as_tensor(
                            np.array(self._actions[batch_inds[i]])[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )
                    batch_next_obs.append(
                        torch.as_tensor(
                            np.array(self._next_obs[batch_inds[i]])[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )
                    batch_rewards.append(
                        torch.as_tensor(
                            np.array(self._rewards[batch_inds[i]], dtype=np.float32)[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )
                    batch_terminateds.append(
                        torch.as_tensor(
                            np.array(self._terminateds[batch_inds[i]])[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )
                    if self._stateful:
                        batch_states.append(
                            torch.as_tensor(
                                np.array(self._states[batch_inds[i]])[
                                    episode_ind_start:episode_ind_end
                                ]
                            )
                        )
                        batch_next_states.append(
                            torch.as_tensor(
                                np.array(self._next_states[batch_inds[i]])[
                                    episode_ind_start:episode_ind_end
                                ]
                            )
                        )
            else:
                for i in range(batch_size):
                    batch_obs.append(
                        torch.as_tensor(np.array(self._obs[batch_inds[i]]))
                    )
                    batch_actions.append(
                        torch.as_tensor(np.array(self._actions[batch_inds[i]]))
                    )
                    batch_next_obs.append(
                        torch.as_tensor(np.array(self._next_obs[batch_inds[i]]))
                    )
                    batch_rewards.append(
                        torch.as_tensor(
                            np.array(self._rewards[batch_inds[i]], dtype=np.float32)
                        )
                    )
                    batch_terminateds.append(
                        torch.as_tensor(np.array(self._terminateds[batch_inds[i]]))
                    )
                    if self._stateful:
                        batch_states.append(
                            torch.as_tensor(
                                np.array(self._states[batch_inds[i]])[
                                    episode_ind_start:episode_ind_end
                                ]
                            )
                        )
                        batch_next_states.append(
                            torch.as_tensor(
                                np.array(self._next_states[batch_inds[i]])[
                                    episode_ind_start:episode_ind_end
                                ]
                            )
                        )

            # Create padded arrays of history
            seq_lengths = torch.LongTensor(list(map(len, batch_obs)))
            batch_obs = pad_sequence(batch_obs).to(self._device)
            batch_actions = torch.unsqueeze(
                pad_sequence(batch_actions).to(self._device), 2
            )
            batch_next_obs = pad_sequence(batch_next_obs).to(self._device)
            batch_rewards = torch.unsqueeze(
                pad_sequence(batch_rewards).to(self._device), 2
            )
            batch_terminateds = torch.unsqueeze(
                pad_sequence(batch_terminateds).to(self._device), 2
            ).long()
            if self._stateful:
                batch_states = pad_sequence(batch_states).to(self._device)
                batch_next_states = pad_sequence(batch_next_states).to(self._device)
        else:
            # Generate non-episodic batch
            for i in range(batch_size):
                batch_obs.append(torch.as_tensor(np.array(self._obs[batch_inds[i]])))
                batch_actions.append(
                    torch.as_tensor(np.array(self._actions[batch_inds[i]]))
                )
                batch_next_obs.append(
                    torch.as_tensor(np.array(self._next_obs[batch_inds[i]]))
                )
                batch_rewards.append(
                    torch.as_tensor(
                        np.array(self._rewards[batch_inds[i]], dtype=np.float32)
                    )
                )
                batch_terminateds.append(
                    torch.as_tensor(np.array(self._terminateds[batch_inds[i]]))
                )
                if self._stateful:
                    batch_states.append(
                        torch.as_tensor(
                            np.array(self._states[batch_inds[i]])[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )
                    batch_next_states.append(
                        torch.as_tensor(
                            np.array(self._next_states[batch_inds[i]])[
                                episode_ind_start:episode_ind_end
                            ]
                        )
                    )

            batch_obs = torch.stack(tuple(batch_obs)).to(self._device)
            batch_actions = torch.unsqueeze(
                torch.stack(tuple(batch_actions)).to(self._device), 1
            )
            batch_next_obs = torch.stack(tuple(batch_next_obs)).to(self._device)
            batch_rewards = torch.unsqueeze(
                torch.stack(tuple(batch_rewards)).to(self._device), 1
            )
            batch_terminateds = torch.unsqueeze(
                torch.stack(tuple(batch_terminateds)).to(self._device), 1
            ).long()
            if self._stateful:
                batch_states = torch.stack(tuple(batch_states)).to(self._device)
                batch_next_states = torch.stack(tuple(batch_next_states)).to(
                    self._device
                )

        # Return batch
        if self._episodic:
            if self._stateful:
                return (
                    batch_obs,
                    batch_actions,
                    batch_next_obs,
                    batch_rewards,
                    batch_terminateds,
                    batch_states,
                    batch_next_states,
                    seq_lengths,
                )
            else:
                return (
                    batch_obs,
                    batch_actions,
                    batch_next_obs,
                    batch_rewards,
                    batch_terminateds,
                    seq_lengths,
                )
        else:
            if self._stateful:
                return (
                    batch_obs,
                    batch_actions,
                    batch_next_obs,
                    batch_rewards,
                    batch_terminateds,
                    batch_states,
                    batch_next_states,
                )
            else:
                return (
                    batch_obs,
                    batch_actions,
                    batch_next_obs,
                    batch_rewards,
                    batch_terminateds,
                )
