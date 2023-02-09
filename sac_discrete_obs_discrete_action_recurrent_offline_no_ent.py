# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import pickle

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from common.models import RecurrentDiscreteActorDiscreteObs, RecurrentDiscreteCriticDiscreteObs
from common.replay_buffer import ReplayBuffer
from common.utils import make_env_gym_pomdp, set_seed, save


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project-name", type=str, default="sac-discrete-obs-discrete-action-recurrent-offline-no-ent",
        help="the wandb's project name")
    parser.add_argument("--wandb-dir", type=str, default="./",
        help="the wandb directory")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="POMDP-heavenhell_1-episodic-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--maximum-episode-length", type=int, default=50,
        help="maximum length for episodes for gym POMDP environment")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--eval-freq", type=int, default=1000,
        help="timestep frequency at which to run evaluation")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")

    # Offline training specific arguments
    parser.add_argument("--dataset-path", type=str, default="heavenhell_1_expert_data.pkl",
        help="path to dataset for training")
    parser.add_argument("--num-evals", type=int, default=10,
        help="number of evaluation episodes to generate per evaluation during training")

    # Checkpointing specific arguments
    parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="checkpoint saving during training")
    parser.add_argument("--save-checkpoint-dir", type=str, default="./trained_models/",
        help="path to directory to save checkpoints in")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
        help="how often to save checkpoints during training (in timesteps)")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to resume training from a checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default="trained_models/POMDP-heavenhell_1-episodic-v0__sac_discrete_obs_discrete_action_recurrent__1__1674880703_3ahou5az/global_step_5000.pth",
        help="path to checkpoint to resume training from")
    parser.add_argument("--run-id", type=str, default=None,
        help="wandb unique run id for resuming")

    args = parser.parse_args()
    # fmt: on
    return args


def eval_policy(
    actor,
    env_name,
    seed,
    seed_offset,
    global_step,
    capture_video,
    run_name,
    max_episode_len,
    num_evals,
    data_log,
):
    # Put actor model in evaluation mode
    actor.eval()

    with torch.no_grad():
        # Initialization
        run_name_full = run_name + "__eval__" + str(global_step)
        env = make_env_gym_pomdp(
            env_name,
            seed + seed_offset,
            0,
            capture_video,
            run_name_full,
            max_episode_len=max_episode_len,
        )

        avg_episodic_return = 0
        avg_episodic_length = 0
        # Start evaluation
        for _ in range(num_evals):
            done = False
            obs = env.reset()
            hidden_in = None
            while not done:
                # Get action
                seq_lengths = torch.LongTensor([1])
                action, _, _, hidden_out = actor.get_action(
                    torch.tensor(obs).to(device).view(1, -1), seq_lengths, hidden_in
                )
                action = action.view(-1).detach().cpu().numpy()[0]
                hidden_in = hidden_out

                # Take step in environment
                next_obs, reward, done, info = env.step(action)

                # Update episodic reward and length
                avg_episodic_return += reward
                avg_episodic_length += 1

                # Update next obs
                obs = next_obs

        avg_episodic_return /= num_evals
        avg_episodic_length /= num_evals
        print(
            f"global_step={global_step}, episodic_return={avg_episodic_return}",
            flush=True,
        )
        data_log["misc/episodic_return"] = avg_episodic_return
        data_log["misc/episodic_length"] = avg_episodic_length

    # Put actor model back in training mode
    actor.train()


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}"
    run_id = wandb.util.generate_id()

    # If a unique wandb run id is given, then resume from that, otherwise
    # generate new run for resuming
    if args.resume and args.run_id is not None:
        wandb.init(
            id=args.run_id,
            dir=args.wandb_dir,
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
            resume="must",
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            group=args.env_id,
            mode="offline",
        )
    else:
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            group=args.env_id,
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
    env = make_env_gym_pomdp(
        args.env_id,
        args.seed,
        0,
        args.capture_video,
        run_name,
        max_episode_len=args.maximum_episode_length,
    )
    # Set RNG state for env
    if args.resume:
        env.np_random.bit_generator.state = checkpoint["rng_states"]["env_rng_state"]
        env.action_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_action_space_rng_state"
        ]
        env.observation_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_obs_space_rng_state"
        ]
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Initialize models and optimizers
    actor = RecurrentDiscreteActorDiscreteObs(env).to(device)
    qf1 = RecurrentDiscreteCriticDiscreteObs(env).to(device)
    qf2 = RecurrentDiscreteCriticDiscreteObs(env).to(device)
    qf1_target = RecurrentDiscreteCriticDiscreteObs(env).to(device)
    qf2_target = RecurrentDiscreteCriticDiscreteObs(env).to(device)
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

    # Load dataset
    dataset = pickle.load(open(args.dataset_path, "rb"))

    # Initialize replay buffer
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        dataset["obs"].shape[0],
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=True,
    )
    rb.load_buffer(dataset)

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    start_global_step = 0
    # If resuming, update starting step
    if args.resume:
        start_global_step = checkpoint["global_step"] + 1
    # ALGO LOGIC: training and evaluation
    for global_step in range(start_global_step, args.total_timesteps):
        # Store values for data logging for each global step
        data_log = {}

        # Sample data from replay buffer
        (
            observations,
            actions,
            next_observations,
            dones,
            rewards,
            seq_lengths,
        ) = rb.sample_history(args.batch_size)
        observations = observations.squeeze(2).long()
        next_observations = next_observations.squeeze(2).long()
        # ---------- update critic ---------- #
        # no grad because target networks are updated separately (pg. 6 of
        # updated SAC paper)
        with torch.no_grad():
            _, next_state_action_probs, next_state_log_pis, _ = actor.get_action(
                next_observations, seq_lengths
            )
            # two Q-value estimates for reducing overestimation bias (pg. 8 of updated SAC paper)
            qf1_next_target = qf1_target(next_observations, seq_lengths)
            qf2_next_target = qf2_target(next_observations, seq_lengths)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            # calculate eq. 3 in updated SAC paper
            qf_next_target = next_state_action_probs * (min_qf_next_target)
            # calculate eq. 2 in updated SAC paper
            next_q_value = rewards + (
                (1 - dones) * args.gamma * qf_next_target.sum(dim=2).unsqueeze(-1)
            )

        # calculate eq. 5 in updated SAC paper
        qf1_a_values = qf1(observations, seq_lengths).gather(2, actions)
        qf2_a_values = qf2(observations, seq_lengths).gather(2, actions)
        q_loss_mask = torch.unsqueeze(
            torch.arange(torch.max(seq_lengths))[:, None] < seq_lengths[None, :], 2
        ).to(device)
        q_loss_mask_nonzero_elements = torch.sum(q_loss_mask).to(device)
        qf1_loss = (
            torch.sum(
                F.mse_loss(qf1_a_values, next_q_value, reduction="none") * q_loss_mask
            )
            / q_loss_mask_nonzero_elements
        )
        qf2_loss = (
            torch.sum(
                F.mse_loss(qf2_a_values, next_q_value, reduction="none") * q_loss_mask
            )
            / q_loss_mask_nonzero_elements
        )
        qf_loss = qf1_loss + qf2_loss

        # calculate eq. 6 in updated SAC paper
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # ---------- update actor ---------- #
        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                _, state_action_probs, state_action_log_pis, _ = actor.get_action(
                    observations, seq_lengths
                )

                # no grad because q-networks are updated separately
                with torch.no_grad():
                    qf1_pi = qf1(observations, seq_lengths)
                    qf2_pi = qf2(observations, seq_lengths)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # calculate eq. 7 in updated SAC paper
                actor_loss_mask = torch.repeat_interleave(
                    q_loss_mask, env.action_space.n, 2
                )
                actor_loss_mask_nonzero_elements = torch.sum(actor_loss_mask)
                actor_loss = state_action_probs * (-min_qf_pi)
                actor_loss = (
                    torch.sum(actor_loss * actor_loss_mask)
                    / actor_loss_mask_nonzero_elements
                )

                # calculate eq. 9 in updated SAC paper
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        # update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )  # "update target network weights" line in page 8, algorithm 1,
                # in updated SAC paper
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        if global_step % 100 == 0:
            data_log["losses/qf1_values"] = qf1_a_values.mean().item()
            data_log["losses/qf2_values"] = qf2_a_values.mean().item()
            data_log["losses/qf1_loss"] = qf1_loss.item()
            data_log["losses/qf2_loss"] = qf2_loss.item()
            data_log["losses/qf_loss"] = qf_loss.item() / 2.0
            data_log["losses/actor_loss"] = actor_loss.item()
            data_log["misc/steps_per_second"] = int(
                global_step / (time.time() - start_time)
            )
            print("SPS:", int(global_step / (time.time() - start_time)), flush=True)

        if (global_step + 1) % args.eval_freq == 0 or global_step == 0:
            eval_policy(
                actor,
                args.env_id,
                args.seed,
                10000,
                global_step,
                args.capture_video,
                run_name,
                args.maximum_episode_length,
                args.num_evals,
                data_log,
            )

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
                # Save replay buffer
                rb_data = {}
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
                    wandb.run.name,
                    run_id,
                    args.save_checkpoint_dir,
                    global_step,
                    models,
                    optimizers,
                    rb_data,
                    rng_states,
                )

    env.close()
