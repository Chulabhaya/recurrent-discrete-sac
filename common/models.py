import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ContinuousCritic(nn.Module):
    """Continuous soft Q-network model for continous SAC."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        """
        Calculates the Q-values of state-action pairs.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Actions.

        Returns
        -------
        q_values : tensor
            Q-values for given state-action pairs.
        """
        x = torch.cat([states, actions], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class ContinuousActor(nn.Module):
    """Continuous actor model for continous SAC."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, states):
        """
        Calculates means and log stds of action distributions generated from input states.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        means : tensor
            Means of action distributions.
        log_stds : tensor
            Log stds of action distributions.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        means = self.fc_mean(x)
        log_stds = self.fc_logstd(x)
        log_stds = torch.tanh(log_stds)
        log_stds = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_stds + 1
        )  # From SpinUp / Denis Yarats

        return means, log_stds

    def get_actions(self, states):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        log_probs : tensor
            Logs of action probabilities, used for entropy.
        means : tensor
            Means of action distributions.
        """
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normal = torch.distributions.Normal(means, stds)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        log_probs = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        means = torch.tanh(means) * self.action_scale + self.action_bias

        return actions, log_probs, means


class RecurrentContinuousCritic(nn.Module):
    """Recurrent continuous soft Q-network model for continous SAC for POMDPs."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions, seq_lengths):
        """
        Calculates the Q-values of state-action pairs.

        Parameters
        ----------
        states : tensor
            States or observations.
        actions : tensor
            Actions.
        seq_lengths : tensor
            Sequence lengths for data in batch.

        Returns
        -------
        q_values : tensor
            Q-values for given state-action pairs.
        """
        x = torch.cat([states, actions], 2)
        x = F.relu(self.fc1(x))

        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class RecurrentContinuousActor(nn.Module):
    """Recurrent continuous actor model for continous SAC for POMDPs."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, states, seq_lengths, in_hidden=None):
        """
        Calculates means and log stds of action distributions generated from input states.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
            Sequence lengths for data in batch.

        Returns
        -------
        means : tensor
            Means of action distributions.
        log_stds : tensor
            Log stds of action distributions.
        out_hidden : tensor
            Hidden layers of LSTM for preserving memory across steps.
        """
        x = F.relu(self.fc1(states))

        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        x = F.relu(self.fc2(x))
        means = self.fc_mean(x)
        log_stds = self.fc_logstd(x)
        log_stds = torch.tanh(log_stds)
        log_stds = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_stds + 1
        )  # From SpinUp / Denis Yarats

        return means, log_stds, out_hidden

    def get_actions(self, states, seq_lengths, in_hidden=None):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
            Sequence lengths for data in batch.
        in_hidden : tensor
            LSTM hidden layer input to use memory from
            prior step.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        log_probs : tensor
            Logs of action probabilities, used for entropy.
        means : tensor
            Means of action distributions.
        out_hidden : tensor
            Hidden layers of LSTM for preserving memory across steps.
        """
        means, log_stds, out_hidden = self(states, seq_lengths, in_hidden)
        stds = log_stds.exp()
        normal = torch.distributions.Normal(means, stds)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        log_probs = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_probs = log_probs.sum(2, keepdim=True)
        means = torch.tanh(means) * self.action_scale + self.action_bias

        return actions, log_probs, means, out_hidden


class RecurrentDiscreteCritic(nn.Module):
    """Recurrent discrete soft Q-network model for discrete SAC for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, states, seq_lengths):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        # Embedding layer
        x = F.relu(self.fc1(states))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class RecurrentDiscreteActor(nn.Module):
    """Recurrent discrete soft actor model for discrete SAC for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states, seq_lengths, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        # Embedding layer
        x = F.relu(self.fc1(states))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_actions(self, states, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Logs of action probabilities, used for entropy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(states, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, out_hidden


class DiscreteCritic(nn.Module):
    """Discrete soft Q-network model for discrete SAC with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, states):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DiscreteActor(nn.Module):
    """Discrete actor model for discrete SAC with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        actions : tensor
            Sampled actions from action distributions.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Logs of action probabilities, used for entropy.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class RecurrentDiscreteCriticDiscreteObs(nn.Module):
    """Recurrent discrete soft Q-network model for POMDPs with discrete actions
    and discrete observations.
    """

    def __init__(self, model_config):
        """Initialize model.

        Args:
            model_config: Dictionary containing model configuration
                parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(model_config["input_size"], 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, model_config["output_size"])

    def forward(self, states, seq_lengths):
        """Calculate Q-values.

        Args:
            states: States or observations.
            seq_lengths: Sequence lengths for data in batch.

        Returns:
            Q-values for actions.
        """
        # Embedding layer
        x = self.embedding(states)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class RecurrentDiscreteActorDiscreteObs(nn.Module):
    """Recurrent discrete actor model for POMDPs with discrete actions
    and discrete observations.
    """

    def __init__(self, model_config):
        """Initialize model.

        Args:
            model_config: Dictionary containing model configuration
                parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(model_config["input_size"], 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, model_config["output_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states, seq_lengths, in_hidden=None):
        """Calculates probabilities for taking each action given a state.

        Args:
            states: States or observations.
            seq_lengths: Sequence lengths for data in batch.
            in_hidden: LSTM hidden layer data.

        Returns:
            Returns a tuple (action_probs, out_hidden), where action_probs
            contains the probabilities of actions, and out_hidden contains
            the LSTM hidden layer data.

        """
        # Embedding layer
        x = self.embedding(states)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_actions(self, states, seq_lengths, in_hidden=None, epsilon=1e-8):
        """Calculates actions.

        Args:
            states: States or observations.
            seq_lengths: Sequence lengths for data in batch.
            in_hidden: LSTM hidden layer data.
            epsilon: Used to ensure no zero probability values.

        Returns:
            Returns a tuple (actions, action_probs, log_action_probs,
            out_hidden), where actions contains the actions sampled from the
            action distribution, action_probs contains the probabilities for
            those actions, log_action_probs contains the log of those
            probabilities, and out_hidden contains LSTM hidden layer data.

        """
        action_probs, out_hidden = self.forward(states, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do
        # log 0
        z = action_probs == 0.0
        z = z.float() * epsilon
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, out_hidden


class DiscreteCriticDiscreteObs(nn.Module):
    """Discrete soft Q-network model with discrete actions and discrete
    observations.
    """

    def __init__(self, model_config):
        """Initialize model.

        Args:
            model_config: Dictionary containing model configuration
                parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(model_config["input_size"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, model_config["output_size"])

    def forward(self, states):
        """Calculate Q-values.

        Args:
            states: States or observations.

        Returns:
            Q-values for actions.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class DiscreteActorDiscreteObs(nn.Module):
    """Discrete actor model with discrete actions and discrete observations.
    """

    def __init__(self, model_config):
        """Initialize model.

        Args:
            model_config: Dictionary containing model configuration
                parameters.
        """
        super().__init__()
        self.embedding = nn.Embedding(model_config["input_size"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, model_config["output_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, states):
        """Calculates probabilities for taking each action given a state.

        Args:
            states: States or observations.

        Returns:
            Action probabilities.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states, epsilon=1e-8):
        """Generates actions.

        Args:
            states: States or observations.
            epsilon: Used to ensure no zero probability values.

        Returns:
            Returns a tuple (actions, action_probs, log_action_probs), where
            actions contains the actions sampled from the action distribution,
            action_probs contains the probabilities for those actions, and
            log_action_probs contains the log of those probabilities.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)

        # Have to deal with situation of 0.0 probabilities because we can't do
        # log 0
        z = action_probs == 0.0
        z = z.float() * epsilon
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
