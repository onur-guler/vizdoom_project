import random
import os
import argparse
import itertools
from datetime import datetime

import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import RecordVideo
import numpy as np
import yaml

import torch
from torch import nn

from cdqn import CDQN
from replay_memory import ReplayMemory, Transition
from typing import List

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from typing import Optional

# For printing date and time
DATE_FORMAT = "%m-%d|%H:%M:%S"

# Directory for saving run info
RUNS_DIR = "run_%m-%d_%H:%M:%S"
RUNS_DIR = datetime.now().strftime(RUNS_DIR)
os.makedirs(RUNS_DIR, exist_ok=True)

# My GPU memory wasn't sufficient, if yours is, use the usual line:
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"


# Deep Q-Learning Agent
class Agent:

    def __init__(
        self,
        hyperparameter_set: str,
        model_filename: Optional[str],
        render_path: Optional[str],
    ):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters["env_id"]
        self.learning_rate      = hyperparameters["learning_rate"]
        self.discount_factor    = hyperparameters["discount_factor"]
        self.network_sync_rate  = hyperparameters["network_sync_rate"]
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size    = hyperparameters["mini_batch_size" ]
        self.epsilon_init       = hyperparameters["epsilon_init"]
        self.epsilon_decay      = hyperparameters["epsilon_decay"]
        self.epsilon_min        = hyperparameters["epsilon_min"]
        self.fc1_nodes          = hyperparameters["fc1_nodes"]
        self.fc2_nodes          = hyperparameters["fc2_nodes"]
        self.env_make_params    = hyperparameters.get("env_make_params", {})

        self.render_episode_count = hyperparameters["render_episode_count"]
        self.mean_n = hyperparameters["mean_n"]
        self.nk_episode_saves = 1000 * hyperparameters["nk_episode_saves"]

        # Neural Network
        self.loss_fn = (nn.MSELoss())
        self.optimizer = None  # NN Optimizer. Initialize later.

        # Path to Run info
        render_dir = RUNS_DIR if render_path is None else render_path
        self.RENDER_DIR = os.path.join(
            render_dir, f"render_{self.hyperparameter_set}"
        )
        self.LOG_FILE = os.path.join(
            RUNS_DIR, f"{self.hyperparameter_set}.log"
        )
        model_filename = (
            f"{self.hyperparameter_set}.pt"
            if model_filename is None
            else model_filename
        )
        self.MODEL_FILE = os.path.join(
            RUNS_DIR if model_filename is None else ".", model_filename
        )
        # TensorBoard log dir
        self.TB_LOGDIR = os.path.join("./", f"tb_{self.hyperparameter_set}")
        os.makedirs(self.TB_LOGDIR, exist_ok=True)

        # SummaryWriter will be created when training starts
        self.writer = None

    def run(
        self,
        is_training: bool = True,
        render: bool = False,
        show_window: bool = False,
    ):
        if is_training:
            start_time = datetime.now()

            log_message = (
                f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            )
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message + "\n")

            # create TensorBoard writer
            # use a run-specific subfolder with timestamp so multiple runs
            # don't clobber each other
            ts = start_time.strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.TB_LOGDIR, ts)
            )

        # Create instance of the environment.
        env = gym.make(
            self.env_id,
            window_visible=show_window,
            render_mode="rgb_array" if render else None,
            **self.env_make_params,
        )

        if render:
            env = RecordVideo(
                env,
                self.RENDER_DIR,
                name_prefix="vizdoom",
                fps=35,
                episode_trigger=lambda x: True,
            )
        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        obs_h, obs_w, obs_d = env.observation_space.shape
        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network.
        policy_net = CDQN(
            action_space=num_actions,
            height=obs_h,
            width=obs_w,
            depth=obs_d,
            fc1_dim=self.fc1_nodes,
            fc2_dim=self.fc2_nodes,
        ).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy
            # network
            target_net = CDQN(
                action_space=num_actions,
                height=obs_h,
                width=obs_w,
                depth=obs_d,
                fc1_dim=self.fc1_nodes,
                fc2_dim=self.fc2_nodes,
            ).to(device)
            target_net.load_state_dict(policy_net.state_dict())

            # Policy network optimizer.
            self.optimizer = torch.optim.Adam(
                policy_net.parameters(), lr=self.learning_rate
            )

            # Track number of steps taken. Used for syncing networks.
            step_count = 0

            # Track best reward
            best_mean_reward = -9999999

        else:
            # Load learned policy
            policy_net.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_net.eval()

        iterate = itertools.count()
        if render:
            iterate = range(self.render_episode_count)
        for episode in iterate:
            state, _ = (env.reset())
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(
                        action, dtype=torch.int64, device=device
                    )
                else:
                    # select best action
                    with torch.no_grad():
                        # Pytorch expects a batch layer, so add batch dimension
                        # argmax finds the index of the largest element.
                        action = policy_net(state.unsqueeze(0)).argmax()

                # Execute action.
                new_state, reward, terminated, truncated, info = env.step(
                    action.item()
                )

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(
                    new_state, dtype=torch.float, device=device
                )
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append(
                        (state, action, new_state, reward, terminated)
                    )
                    step_count += 1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Log episode reward to TensorBoard (scalar)
            if is_training and self.writer:
                self.writer.add_scalar(
                    "Episode/Reward", episode_reward, global_step=episode
                )
                # Log rolling mean reward
                mean_reward = float(
                    np.mean(
                        rewards_per_episode[
                            max(0, len(rewards_per_episode) - self.mean_n):
                        ]
                    )
                )
                self.writer.add_scalar(
                    f"Episode/MeanReward{self.mean_n}",
                    mean_reward,
                    global_step=episode,
                )

            # Save model when new best reward is obtained.
            if is_training:
                if episode % self.mean_n:
                    mean_reward = float(
                        np.mean(
                            rewards_per_episode[
                                max(
                                    0, len(rewards_per_episode) - self.mean_n
                                ):
                            ]
                        )
                    )
                    if (
                        mean_reward > best_mean_reward
                        and len(rewards_per_episode) >= self.mean_n
                    ):
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}"
                        log_message += f": New best mean reward {mean_reward} "
                        log_message += f"at episode {episode}, saving model"
                        print(log_message)
                        with open(self.LOG_FILE, "a") as file:
                            file.write(log_message + "\n")

                        torch.save(policy_net.state_dict(), self.MODEL_FILE)
                        best_mean_reward = mean_reward
                if episode % self.nk_episode_saves == 0:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: "
                    log_message += f"episode {episode} reached, saving model"
                    print(log_message)
                    torch.save(
                        policy_net.state_dict(),
                        os.path.join(
                            RUNS_DIR,
                            f"{episode // 1000}k_{self.hyperparameter_set}.pt",
                        ),
                    )

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    loss = self.optimize(mini_batch, policy_net, target_net)

                    # Log training loss and epsilon to TensorBoard
                    if self.writer:
                        loss_val = loss.item()
                        self.writer.add_scalar(
                            "Train/Loss", loss_val, global_step=episode
                        )
                        self.writer.add_scalar(
                            "Train/Epsilon", epsilon, global_step=episode
                        )
                        # Decay epsilon
                        epsilon = max(
                            epsilon * self.epsilon_decay, self.epsilon_min
                        )

                    # Copy policy network to target network after a certain
                    # number of steps
                    if step_count > self.network_sync_rate:
                        target_net.load_state_dict(policy_net.state_dict())
                        step_count = 0

    # Optimize policy network
    def optimize(
        self, mini_batch: List[Transition], policy_net: CDQN, target_net: CDQN
    ):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            next_actions = target_net(new_states).argmax(dim=1, keepdims=True)
            q_next = target_net(new_states).gather(1, next_actions).squeeze()
            target_q = (
                rewards + (1 - terminations) * self.discount_factor * q_next
            )
            target_q = target_q.squeeze()

        # Calculate Q values from current policy
        current_q = (
            policy_net(states)
            .gather(dim=1, index=actions.unsqueeze(dim=1))
            .squeeze()
        )

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network parameters, weights and biases

        return loss.detach()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters")
    parser.add_argument(
        "--train", "-t", help="Training mode", action="store_true"
    )
    parser.add_argument("--model-file", "-f")
    parser.add_argument(
        "-w", "--show-window", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-r", "--render-video", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("-o", "--render-path")

    args = parser.parse_args()

    dql = Agent(
        hyperparameter_set=args.hyperparameters,
        model_filename=args.model_file,
        render_path=args.render_path,
    )

    if args.train:
        dql.run(
            is_training=True,
            render=args.render_video,
            show_window=args.show_window,
        )
    else:
        dql.run(
            is_training=False,
            render=args.render_video,
            show_window=args.show_window,
        )
