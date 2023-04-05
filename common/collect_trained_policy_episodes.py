import argparse
import pickle
from distutils.util import strtobool
import random

import torch

from models import RecurrentDiscreteActorDiscreteObs
from replay_buffer import EpisodicReplayBuffer
from utils import make_env, set_seed


def collect_trained_policy_data(env, actor, device, seed, total_timesteps, epsilon):
    # Initialize replay buffer for storing data
    rb = EpisodicReplayBuffer(
        total_timesteps,
        device,
    )

    # Initialize environment interaction loop
    terminated, truncated = False, False
    hidden_in = None
    obs, info = env.reset(seed=seed)

    # Generate data
    episodic_count = 0
    global_step = 0
    for global_step in range(0, total_timesteps):
        # Take either random or scripted action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            seq_lengths = torch.LongTensor([1])
            action, _, _, hidden_out = actor.get_actions(
                torch.tensor(obs).to(device).view(1, -1), seq_lengths, hidden_in
            )
            action = action.view(-1).detach().cpu().numpy()[0]
            hidden_in = hidden_out

        # Take action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Save data to replay buffer
        rb.add(obs, action, next_obs, reward, terminated, truncated)

        # Update next obs
        obs = next_obs

        # If episode is over, reset environment and
        # update episode count
        if terminated or truncated:
            print(
                f"global_step={global_step}, episodic_return={info['episode']['r'][0]}, episodic_length={info['episode']['l'][0]}",
                flush=True,
            )
            episodic_count += 1
            hidden_in = None
            obs, info = env.reset()

    return rb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Dataset generation params
    parser.add_argument("--seed", type=int, default=322,
        help="seed of data generation")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--env-id", type=str, default="POMDP-heavenhell_1-episodic-v0",
        help="the id of the environment for the trained policy")
    parser.add_argument("--maximum-episode-length", type=int, default=50,
        help="maximum length for episodes for gym POMDP environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of data to gather from policy")
    parser.add_argument("--checkpoint", type=str, default="/home/chulabhaya/phd/research/data/heavenhell_1/3-23-23_heavenhell_1_sac_expert_policy.pth",
        help="path to checkpoint with trained policy")
    parser.add_argument("--epsilon", type=float, default=1.0,
        help="random action sampling percentage")

    args = parser.parse_args()
    # fmt: on
    return args


def main():
    args = parse_args()

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Running on the following device: " + device.type, flush=True)

    # Set seeds
    set_seed(args.seed, device)

    # Initialize environment for generating dataset
    env = make_env(
        args.env_id,
        args.seed,
        False,
        "",
        max_episode_len=args.maximum_episode_length,
    )

    # Load trained policy
    print("Loading policy from checkpoint: " + args.checkpoint, flush=True)
    checkpoint = torch.load(args.checkpoint)

    # Initialize actor/policy
    actor = RecurrentDiscreteActorDiscreteObs(env).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"]["actor_state_dict"])

    # Collect dataset in a replay buffer
    rb = collect_trained_policy_data(
        env, actor, device, args.seed, args.total_timesteps, args.epsilon
    )

    # Get dictionary of replay buffer data
    rb_data = rb.save_buffer()

    # Save out dictionary into pickle file
    f = open("3-23-23_heavenhell_1_sac_expert_policy_100_percent_random_data.pkl", "wb")
    pickle.dump(rb_data, f)
    f.close()


if __name__ == "__main__":
    main()
