# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from models import (RecurrentDiscreteActorDiscreteObs,
                    RecurrentDiscreteCriticDiscreteObs)
from replay_buffer import ReplayBuffer
from utils import make_env_gym_pomdp, set_seed, save


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project-name", type=str, default="sac-discrete-obs-discrete-action-recurrent-offline",
        help="the wandb's project name")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="POMDP-heavenhell_1-episodic-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
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
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def eval_policy(
    actor, env_name, seed, seed_offset, global_step, capture_video, run_name, data_log
):
    with torch.no_grad():
        # Initialization
        run_name_full = run_name + "__eval__" + str(global_step)
        envs = gym.vector.SyncVectorEnv(
            [make_env_gym_pomdp(env_name, seed + seed_offset, 0, capture_video, run_name_full)]
        )

        done = False
        obs = envs.reset()
        hidden_in = None

        # TODO: Run a series of 10 evaluations each time this function is called and
        # average the results for returning back

        # Start evaluation
        timestep = 0
        while not done:
            # Get action
            seq_lengths = torch.LongTensor([1])
            actions, _, _, hidden_out = actor.get_action(
                torch.unsqueeze(torch.tensor(obs).to(device), 0), seq_lengths, hidden_in
            )
            actions = torch.squeeze(actions, 0).detach().cpu().numpy()
            hidden_in = hidden_out

            # Step through environment
            next_obs, rewards, dones, infos = envs.step(actions)

            print(
                "Obs: {}, Action: {}, Next Obs: {}, Reward: {}, Dones: {}".format(
                    obs, actions, next_obs, rewards, dones
                )
            )

            # Plotting returns
            for info in infos:
                if "episode" in info.keys():
                    print(
                        f"global_step={global_step}, eval_episodic_return={info['episode']['r']}"
                    )
                    data_log["misc/eval_episodic_return"] = info["episode"]["r"]
                    data_log["misc/eval_episodic_length"] = info["episode"]["r"]
                    break

            # Check for termination
            for idx, done in enumerate(dones):
                if done:
                    break
                else:
                    timestep += 1
                    obs = next_obs
                    print(timestep)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project=args.wandb_project_name,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env_gym_pomdp(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Initialize models and optimizers
    actor = RecurrentDiscreteActorDiscreteObs(envs).to(device)
    qf1 = RecurrentDiscreteCriticDiscreteObs(envs).to(device)
    qf2 = RecurrentDiscreteCriticDiscreteObs(envs).to(device)
    qf1_target = RecurrentDiscreteCriticDiscreteObs(envs).to(device)
    qf2_target = RecurrentDiscreteCriticDiscreteObs(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -0.3 * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    # Load dataset
    dataset = np.load("heavenhell_1_1000_episode_dataset_random_100percent.npz")
    dataset_obs = dataset["observations"]
    dataset_actions = dataset["actions"]
    dataset_next_obs = dataset["next_observations"]
    dataset_dones = dataset["dones"]
    dataset_rewards = dataset["rewards"]
    dataset_timeouts = dataset["timeouts"]
    buffer_size = dataset_obs.size

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    rb.add_dataset(
        dataset_obs,
        dataset_next_obs,
        dataset_actions,
        dataset_rewards,
        dataset_dones,
        dataset_timeouts,
    )
    start_time = time.time()

    # ALGO LOGIC: training and evaluation
    for global_step in range(args.total_timesteps):
        # Store values for data logging for each global step
        data_log = {}

        # sample data from replay buffer
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
            qf_next_target = next_state_action_probs * (
                min_qf_next_target - alpha * next_state_log_pis
            )
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
        # clip_grad_norm_(qf1.parameters(), 1)
        # clip_grad_norm_(qf2.parameters(), 1)
        q_optimizer.step()

        # ---------- update actor ---------- #
        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                _, state_action_probs, state_action_log_pis, _ = actor.get_action(
                    observations, seq_lengths
                )
                qf1_pi = qf1(observations, seq_lengths)
                qf2_pi = qf2(observations, seq_lengths)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                # calculate eq. 7 in updated SAC paper
                actor_loss_mask = torch.repeat_interleave(q_loss_mask, envs.single_action_space.n, 2)
                actor_loss_mask_nonzero_elements = torch.sum(actor_loss_mask)
                actor_loss = state_action_probs * (
                    (alpha * state_action_log_pis) - min_qf_pi
                )
                actor_loss = (
                    torch.sum(actor_loss * actor_loss_mask)
                    / actor_loss_mask_nonzero_elements
                )

                # calculate eq. 9 in updated SAC paper
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ---------- update alpha ---------- #
                if args.autotune:
                    with torch.no_grad():
                        (
                            _,
                            state_action_probs,
                            state_action_log_pis,
                            _,
                        ) = actor.get_action(observations, seq_lengths)
                    # calculate eq. 18 in updated SAC paper
                    alpha_loss = state_action_probs * (
                        -log_alpha * (state_action_log_pis + target_entropy)
                    )
                    alpha_loss = (
                        torch.sum(alpha_loss * actor_loss_mask)
                        / actor_loss_mask_nonzero_elements
                    )

                    # calculate gradient of eq. 18
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

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
            data_log["losses/alpha"] = alpha
            data_log["misc/steps_per_second"] = int(
                global_step / (time.time() - start_time)
            )
            print("SPS:", int(global_step / (time.time() - start_time)))
            if args.autotune:
                data_log["losses/alpha_loss"] = alpha_loss.item()

        if (global_step + 1) % args.eval_freq == 0 or global_step == 0:
            eval_policy(
                actor,
                args.env_id,
                args.seed,
                100,
                global_step,
                args.capture_video,
                run_name,
                data_log,
            )

        data_log["misc/global_step"] = global_step
        wandb.log(data_log)

    envs.close()
