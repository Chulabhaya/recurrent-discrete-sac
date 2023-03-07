import string
import gymnasium as gym
import numpy as np
import random
import gymnasium_pomdps


class HeavenHell2ScriptedPolicy:
    def __init__(self) -> None:
        self.heaven_side = None

    def get_action(self, observation: int, prev_done: bool) -> int:
        """Returns the optimal action given an observation from
        a HeavenHell environment.

        Parameters
        ----------
        observation : int
            The current observation.
        prev_done : bool
            The previous done condition, used to reset the
            heaven side if the previous episode ended.

        Returns
        -------
        action : int
            The next action to take.
        """
        # Reset heaven side the episode ended
        if prev_done:
            self.heaven_side = None
        # Take different actions depending on whether
        # we know which side heaven is on yet or not

        # If we don't know which side heaven is on, then
        # all actions should be leading towards the priest
        if self.heaven_side is None:
            if observation == 0:
                action = 1
            elif observation == 1:
                action = 1
            elif observation == 2:
                action = 1
            elif observation == 3:
                action = 2
            elif observation == 4:
                action = 2
            elif observation == 5:
                action = 3
            elif observation == 6:
                action = 3
            elif observation == 7:
                action = 2
            elif observation == 8:
                action = 2
            elif observation == 9:
                self.heaven_side = "left"
                action = 3
            elif observation == 10:
                self.heaven_side = "right"
                action = 3
            elif observation == 11:
                action = 1
            else:
                raise Exception("Unrecognized observation provided!")
        # Once we know which side heaven is on, then
        # all actions should lead towards it
        # We don't consider actions for the priest
        # observations since we already know the side
        # and took an action from above logic
        elif self.heaven_side == "left":
            if observation == 0:
                action = 0
            elif observation == 1:
                action = 0
            elif observation == 2:
                action = 3
            elif observation == 3:
                action = 3
            elif observation == 4:
                action = 3
            elif observation == 5:
                action = 3
            elif observation == 6:
                action = 3
            elif observation == 7:
                action = 0
            elif observation == 8:
                action = 3
            else:
                raise Exception("Unrecognized observation provided!")
        elif self.heaven_side == "right":
            if observation == 0:
                action = 0
            elif observation == 1:
                action = 0
            elif observation == 2:
                action = 2
            elif observation == 3:
                action = 2
            elif observation == 4:
                action = 2
            elif observation == 5:
                action = 2
            elif observation == 6:
                action = 2
            elif observation == 7:
                action = 0
            elif observation == 8:
                action = 3
            else:
                raise Exception("Unrecognized observation provided!")
        else:
            raise Exception("Unrecognized status for the side of heaven!")

        return action


class HeavenHell1ScriptedPolicy:
    def __init__(self) -> None:
        self.heaven_side = None

    def get_action(self, env, observation: int, prev_done: bool, epsilon: float) -> int:
        """Returns the optimal action given an observation from
        a HeavenHell environment.

        Parameters
        ----------
        env : Gym environment
            Gym environment to be used for sampling action space.
        observation : int
            The current observation.
        prev_done : bool
            The previous done condition, used to reset the
            heaven side if the previous episode ended.
        epsilon: float
            Probability of taking a random action.

        Returns
        -------
        action : int
            The next action to take.
        """
        # Reset heaven side the episode ended
        if prev_done:
            self.heaven_side = None
        # Take different actions depending on whether
        # we know which side heaven is on yet or not

        # If we don't know which side heaven is on, then
        # all actions should be leading towards the priest
        if self.heaven_side is None:
            if observation == 0:
                action = 1
            elif observation == 1:
                action = 1
            elif observation == 4:
                action = 2
            elif observation == 3:
                action = 3
            elif observation == 2:
                action = 2
            elif observation == 5:
                self.heaven_side = "left"
                action = 3
            elif observation == 6:
                self.heaven_side = "right"
                action = 3
            elif observation == 7:
                action = 1
            else:
                raise Exception("Unrecognized observation provided!")
        # Once we know which side heaven is on, then
        # all actions should lead towards it
        # We don't consider actions for the priest
        # observations since we already know the side
        # and took an action from above logic
        elif self.heaven_side == "left":
            if observation == 4:
                action = 0
            elif observation == 0:
                action = 0
            elif observation == 1:
                action = 3
            elif observation == 3:
                action = 3
            elif observation == 2:
                action = 3
            elif observation == 5:
                action = 3
            elif observation == 6:
                action = 3
            else:
                raise Exception("Unrecognized observation provided!")
        elif self.heaven_side == "right":
            if observation == 4:
                action = 0
            elif observation == 0:
                action = 0
            elif observation == 1:
                action = 2
            elif observation == 3:
                action = 2
            elif observation == 2:
                action = 2
            elif observation == 5:
                action = 3
            elif observation == 6:
                action = 3
            else:
                raise Exception("Unrecognized observation provided!")
        else:
            raise Exception("Unrecognized status for the side of heaven!")

        # Take random action based on epsilon probability
        if epsilon > 0:
            if random.random() < epsilon:
                action = env.action_space.sample()

        return action


def collect_episodes_scripted_policy(
    environment: string, num_episodes: int, max_episode_steps: int, epsilon: float
) -> dict:
    """Returns dataset from provided POMDP environment.

    Parameters
    ----------
    environment : string
        The POMDP environment.
    num_episodes : int
        How many episodes to collect from the environment.
    max_episode_steps : int
        Maximum length of episodes in timesteps
    epsilon : float
        Probability of taking a random action

    Returns
    -------
    dataset : dict
        Dataset of episodes.
    """
    # Generate environment
    env = gym.wrappers.TimeLimit(
        gym.make(environment),
        max_episode_steps=max_episode_steps,
    )

    # Initialize agent for scripted policy
    if environment == "POMDP-heavenhell_2-episodic-v0":
        agent = HeavenHell2ScriptedPolicy()
    elif environment == "POMDP-heavenhell_1-episodic-v0":
        agent = HeavenHell1ScriptedPolicy()

    # Episode tracker
    episode_count = 0

    # Episode storage
    observation_dataset = []
    next_observation_dataset = []
    action_dataset = []
    reward_dataset = []
    terminated_dataset = []
    truncated_dataset = []
    timeout_dataset = []

    # Initialize environment interaction loop
    terminated, truncated = False, False
    done = False
    obs, info = env.reset()

    # Generate episodes
    episode_timestep = 0
    while episode_count < num_episodes:
        # Get action from scripted policy
        action = agent.get_action(env, obs, done, epsilon)

        # Take action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update dataset
        observation_dataset.append(obs)
        next_observation_dataset.append(next_obs)
        action_dataset.append(action)
        reward_dataset.append(reward)
        terminated_dataset.append(terminated)
        truncated_dataset.append(truncated)

        # If episode is over, reset environment and
        # update episode count
        if terminated or truncated:
            obs, info = env.reset()
            episode_count += 1
            done = True
        else:
            obs = next_obs
            episode_timestep += 1

    # Return datasets
    observation_dataset = np.array(observation_dataset)
    next_observation_dataset = np.array(next_observation_dataset)
    action_dataset = np.array(action_dataset)
    reward_dataset = np.array(reward_dataset)
    terminated_dataset = np.array(terminated_dataset)
    truncated_dataset = np.array(truncated_dataset)

    combined_dataset = {
        "observations": observation_dataset,
        "actions": action_dataset,
        "next_observations": next_observation_dataset,
        "rewards": reward_dataset,
        "terminateds": terminated_dataset,
        "truncateds": truncated_dataset,
        "timeouts": timeout_dataset,
    }

    return combined_dataset


def main():
    dataset_1000_episodes = collect_episodes_scripted_policy(
        "POMDP-heavenhell_1-episodic-v0", 1000, 50, 0.0
    )
    np.savez(
        "heavenhell_1_1000_episode_dataset_random_0percent.npz",
        observations=dataset_1000_episodes["observations"],
        actions=dataset_1000_episodes["actions"],
        next_observations=dataset_1000_episodes["next_observations"],
        rewards=dataset_1000_episodes["rewards"],
        terminateds=dataset_1000_episodes["terminateds"],
        truncateds=dataset_1000_episodes["truncateds"],
    )


if __name__ == "__main__":
    main()
