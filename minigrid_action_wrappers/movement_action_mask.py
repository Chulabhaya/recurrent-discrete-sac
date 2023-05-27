from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces


class Actions(IntEnum):
    """
    Action space for this environment.
    """

    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class MovementActionMask(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Wrapper that restricts MiniGrid action space to only left/0, right/1, forward/2."""

    def __init__(self, env: gym.Env):
        """
        Initialize wrapper.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action):
        """Restricts actions.

        Args:
            action: The action to map

        Returns:
            The same action (no mapping necessary)
        """
        return action
