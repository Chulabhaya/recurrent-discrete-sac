import argparse
import pickle
from distutils.util import strtobool

import torch

from models import RecurrentDiscreteActorDiscreteObs
from replay_buffer import ReplayBuffer
from utils import make_env_gym_pomdp, set_seed


def collect_trained_policy_data(env, actor, device, total_timesteps):
    # Initialize replay buffer for storing data
    rb = ReplayBuffer(
        total_timesteps,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=True,
    )

    # Initialize environment interaction loop
    done = False
    hidden_in = None
    obs = env.reset()

    # Generate data
    episodic_return = 0
    episodic_length = 0
    episodic_count = 0
    global_step = 0
    for global_step in range(0, total_timesteps):
        # Get action from scripted policy
        seq_lengths = torch.LongTensor([1])
        action, _, _, hidden_out = actor.get_action(
            torch.tensor(obs).to(device).view(1, -1), seq_lengths, hidden_in
        )
        action = action.view(-1).detach().cpu().numpy()[0]
        hidden_in = hidden_out

        # Take action in environment
        next_obs, reward, done, info = env.step(action)

        # Save data to replay buffer
        rb.add(obs, next_obs, action, reward, done, info)

        # Update next obs
        obs = next_obs

        # Update episodic reward and length
        episodic_return += reward
        episodic_length += 1

        # If episode is over, reset environment and
        # update episode count
        if done:
            print(
                f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}, episodic_count={episodic_count}",
                flush=True,
            )
            episodic_count += 1
            episodic_return = 0
            episodic_length = 0
            hidden_in = None
            obs = env.reset()

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
    parser.add_argument("--total-timesteps", type=int, default=10000,
        help="total timesteps of data to gather from policy")
    parser.add_argument("--checkpoint", type=str, default="trained_models/POMDP-heavenhell_1-episodic-v0__laptop_seed1_run1_cudnn_on__1__1675040882_3d0707f5/global_step_100000.pth",
        help="path to checkpoint with trained policy")

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
    env = make_env_gym_pomdp(
        args.env_id,
        args.seed,
        0,
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
    rb = collect_trained_policy_data(env, actor, device, args.total_timesteps)

    # Get dictionary of replay buffer data
    rb_data = rb.save_buffer()

    # Save out dictionary into pickle file
    # create a binary pickle file
    f = open("test_data.pkl", "wb")
    pickle.dump(rb_data, f)
    f.close()


if __name__ == "__main__":
    main()
