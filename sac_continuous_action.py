import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from common.models import ContinuousActor, ContinuousCritic
from common.replay_buffer import ReplayBuffer
from common.utils import make_env, save, set_seed


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--exp-group", type=str, default=None,
        help="the group under which this experiment falls")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project", type=str, default="sac-continuous-action",
        help="wandb project name")
    parser.add_argument("--wandb-dir", type=str, default="./",
        help="the wandb directory")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPoleContinuous-P-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")

    # Checkpointing specific arguments
    parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="checkpoint saving during training")
    parser.add_argument("--save-checkpoint-dir", type=str, default="./trained_models/",
        help="path to directory to save checkpoints in")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
        help="how often to save checkpoints during training (in timesteps)")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to resume training from a checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default=None,
        help="path to checkpoint to resume training from")
    parser.add_argument("--run-id", type=str, default=None,
        help="wandb unique run id for resuming")

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"
    wandb_id = wandb.util.generate_id()
    run_id = f"{run_name}_{wandb_id}"

    # If a unique wandb run id is given, then resume from that, otherwise
    # generate new run for resuming
    if args.resume and args.run_id is not None:
        run_id = args.run_id
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            resume="must",
            mode="offline",
        )
    else:
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            mode="offline",
        )

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Running on the following device: " + device.type, flush=True)

    # Set seeding
    set_seed(args.seed, device)

    # Load checkpoint if resuming
    if args.resume:
        print("Resuming from checkpoint: " + args.resume_checkpoint_path, flush=True)
        checkpoint = torch.load(args.resume_checkpoint_path)

    # Set RNG state for seeds if resuming
    if args.resume:
        random.setstate(checkpoint["rng_states"]["random_rng_state"])
        np.random.set_state(checkpoint["rng_states"]["numpy_rng_state"])
        torch.set_rng_state(checkpoint["rng_states"]["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["rng_states"]["torch_cuda_rng_state"])
            torch.cuda.set_rng_state_all(
                checkpoint["rng_states"]["torch_cuda_rng_state_all"]
            )

    # Env setup
    env = make_env(args.env_id, args.seed)
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(env.action_space.high[0])

    actor = ContinuousActor(env).to(device)
    qf1 = ContinuousCritic(env).to(device)
    qf2 = ContinuousCritic(env).to(device)
    qf1_target = ContinuousCritic(env).to(device)
    qf2_target = ContinuousCritic(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # If resuming training, load models and optimizers
    if args.resume:
        actor.load_state_dict(checkpoint["model_state_dict"]["actor_state_dict"])
        qf1.load_state_dict(checkpoint["model_state_dict"]["qf1_state_dict"])
        qf2.load_state_dict(checkpoint["model_state_dict"]["qf2_state_dict"])
        qf1_target.load_state_dict(
            checkpoint["model_state_dict"]["qf1_target_state_dict"]
        )
        qf2_target.load_state_dict(
            checkpoint["model_state_dict"]["qf2_target_state_dict"]
        )
        q_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["q_optimizer"])
        actor_optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]["actor_optimizer"]
        )

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        if args.resume:
            log_alpha = checkpoint["model_state_dict"]["log_alpha"]
        else:
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        # If resuming, load optimizer
        if args.resume:
            a_optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]["a_optimizer"]
            )

        alpha = log_alpha.exp().item()
    else:
        alpha = args.alpha

    # Initialize replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        episodic=False,
        stateful=False,
        device=device,
    )
    # If resuming training, then load previous replay buffer
    if args.resume:
        rb_data = checkpoint["replay_buffer"]
        rb.load_buffer(rb_data)

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    start_global_step = 0
    # If resuming, update starting step
    if args.resume:
        start_global_step = checkpoint["global_step"] + 1

    obs, info = env.reset(seed=args.seed)
    # Set RNG state for env
    if args.resume:
        env.np_random.bit_generator.state = checkpoint["rng_states"]["env_rng_state"]
        env.action_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_action_space_rng_state"
        ]
        env.observation_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_obs_space_rng_state"
        ]
    for global_step in range(start_global_step, args.total_timesteps):
        # Store values for data logging for each global step
        data_log = {}

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = actor.get_actions(
                torch.tensor(obs, dtype=torch.float32).to(device)
            )
            action = action.detach().cpu().numpy()

        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Save data to replay buffer
        rb.add(obs, action, next_obs, reward, terminated, truncated)

        # Update next obs
        obs = next_obs

        # Handle episode end, record rewards for plotting purposes
        if terminated or truncated:
            print(
                f"global_step={global_step}, episodic_return={info['episode']['r'][0]}, episodic_length={info['episode']['l'][0]}",
                flush=True,
            )
            data_log["misc/episodic_return"] = info["episode"]["r"][0]
            data_log["misc/episodic_length"] = info["episode"]["l"][0]

            obs, info = env.reset()

        # ALGO LOGIC: training
        if global_step > args.learning_starts:
            # sample data from replay buffer
            (
                observations,
                actions,
                next_observations,
                rewards,
                terminateds,
            ) = rb.sample(args.batch_size)
            # no grad because target networks are updated separately (pg. 6 of
            # updated SAC paper)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_actions(
                    next_observations
                )
                # two Q-value estimates for reducing overestimation bias (pg. 8 of updated SAC paper)
                qf1_next_target_values = qf1_target(
                    next_observations, next_state_actions
                )
                qf2_next_target_values = qf2_target(
                    next_observations, next_state_actions
                )
                # calculate eq. 3 in updated SAC paper
                min_qf_next_target_values = (
                    torch.min(qf1_next_target_values, qf2_next_target_values)
                    - alpha * next_state_log_pi
                )
                # calculate eq. 2 in updated SAC paper
                next_q_values = rewards.flatten() + (
                    1 - terminateds.flatten()
                ) * args.gamma * (min_qf_next_target_values).view(-1)

            # calculate eq. 5 in updated SAC paper
            qf1_a_values = qf1(observations, actions).view(-1)
            qf2_a_values = qf2(observations, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_values)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_values)
            qf_loss = qf1_loss + qf2_loss

            # calculate eq. 6 in updated SAC paper
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_actions(observations)

                    # no grad because q-networks are updated separately
                    with torch.no_grad():
                        qf1_pi = qf1(observations, pi)
                        qf2_pi = qf2(observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

                    # calculate eq. 7 in updated SAC paper
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    # calculate eq. 9 in updated SAC paper
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        # no grad because actor network is updated separately
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_actions(observations)
                        # calculate eq. 18 in updated SAC paper
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        # calculate gradient of eq. 18
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )  # "update target network weights" line in page 8, algorithm 1,
                    # in updated SAC paper
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                data_log["losses/qf1_values"] = qf1_a_values.mean().item()
                data_log["losses/qf2_values"] = qf2_a_values.mean().item()
                data_log["losses/qf1_loss"] = qf1_loss.item()
                data_log["losses/qf2_loss"] = qf2_loss.item()
                data_log["losses/qf_loss"] = qf_loss.item()
                data_log["losses/actor_loss"] = actor_loss.item()
                data_log["losses/alpha"] = alpha
                data_log["misc/steps_per_second"] = int(
                    global_step / (time.time() - start_time)
                )
                print("SPS:", int(global_step / (time.time() - start_time)), flush=True)
                if args.autotune:
                    data_log["losses/alpha_loss"] = alpha_loss.item()

        data_log["misc/global_step"] = global_step
        wandb.log(data_log, step=global_step)

        # Save checkpoints during training
        if args.save:
            if global_step % args.checkpoint_interval == 0:
                # Save models
                models = {
                    "actor_state_dict": actor.state_dict(),
                    "qf1_state_dict": qf1.state_dict(),
                    "qf2_state_dict": qf2.state_dict(),
                    "qf1_target_state_dict": qf1_target.state_dict(),
                    "qf2_target_state_dict": qf2_target.state_dict(),
                }
                # Save optimizers
                optimizers = {
                    "q_optimizer": q_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                }
                if args.autotune:
                    optimizers["a_optimizer"] = a_optimizer.state_dict()
                    models["log_alpha"] = log_alpha
                # Save replay buffer
                rb_data = rb.save_buffer()
                # Save random states, important for reproducibility
                rng_states = {
                    "random_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "env_rng_state": env.np_random.bit_generator.state,
                    "env_action_space_rng_state": env.action_space.np_random.bit_generator.state,
                    "env_obs_space_rng_state": env.observation_space.np_random.bit_generator.state,
                }
                if device.type == "cuda":
                    rng_states["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
                    rng_states[
                        "torch_cuda_rng_state_all"
                    ] = torch.cuda.get_rng_state_all()

                save(
                    run_id,
                    args.save_checkpoint_dir,
                    global_step,
                    models,
                    optimizers,
                    rb_data,
                    rng_states,
                )

    env.close()
