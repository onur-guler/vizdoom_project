# VizDoom Visual Survival: CMP4501 Project by Onur Sinan Güler
![evo](https://github.com/user-attachments/assets/6a3f9f22-f7e5-4182-b98c-1a1cbc70b2d4)


# Running from host
```sh
python -m venv venv # I prefer venv you can use conda etc. if you want to
source venv/bin/activate
pip install -r requirements.txt
python main.py <args>
```
> [!NOTE]
> Never versions of gcc can't compile ZDoom set CC environment variable if needed
> Ex: `CC=gcc-14 pip install -r requirements.txt`

# Running from Docker/Podman
```sh
docker build -t vizdoom_project .
docker run -v .:/usr/src/project/ -i --gpus=all vizdoom_project <args>
```

# Before development run this command to install pre-commit hooks
```sh
pip install pre-commit && pre-commit install
```

# Project description:
## Visual Survival (ViZDoom)
 
The Goal: Train an agent to survive in a 3D environment, managing health and ammo.
The Challenge: Computer Vision. Your agent must learn from raw pixels (using CNNs), not just coordinate numbers.
Library: vizdoom
Documentation: https://vizdoom.farama.org/

---

Before starting any reinforcement learning stuff, I created my own environment and recording methods to see how I can apply this to other problems and get familiar with python. After I had a pretty good grasp on python, its common libraries like numpy, scipi, tqdm etc., type hints, and vizdoom itself, I created a [custom gymnasium environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation) for this project. This environment can be found at `./gymnasium_env`. The only parts that I couldn't find in the documentations and was on my old environment was how I could render a bgr array instead of rgb and putting audio buffers to the render. But those weren't that important for the project so I just converted the bgr images to rgb and didn't add audio to the rendered video.  

And here's our glorious random agent:

https://github.com/user-attachments/assets/efe31756-529c-45e7-a342-488b03816b54



Since this was my first time ever doing anything about neural networks and reinforcement learning I preferred using the basic scenario.  
The basic scenario consists of 3 actions (MOVE_LEFT, MOVE_RIGHT, and ATTACK) with a cacodemon (The red enemy with a single green eye) that gets killed by a single shot from our pistol. Our rewards are set to, +106 for killing the monster, -5 for every shot, and -1 living reward.  

So the reward function is as follows:


$$
R = 
\begin{cases} 
106 & \text{if monster is killed} \\
-5 \cdot n_{\text{shots}} & \text{for every shot taken} \\
-1 & \text{for each time step (living reward)}
\end{cases}
$$

Since the assignment wants us to learn from raw pixel data by using a CNN, DQNs are suitable for this problem
Deep Q Networks combine reinforcement learning with deep neural networks and learn the optimal policy by approximating the Q values instead of creating a Q table.
The state we get after the preprocessing is huge with 90000 features (3 channel 150x200 picture). I could've used a gray8 picture with a lower resolution to resolve this but I've decided that I'll use this state representation since it will give the agent more data to work with.  

Before using convolutional deep Q networks I tried both Q tables and deep Q learning, but Q tables used way too much memory and deep Q learning didn't give the results that I wished for.

The convolutional deep Q network (In file `./cdqn.py`):
<details>
<summary>Code</summary>

```py
import torch
from torch import nn

class CDQN(nn.Module):
    def __init__(
        self,
        action_space: int,
        width: int,
        height: int,
        depth: int,
        fc1_dim: int = 512,
        fc2_dim: int = 512,
    ):
        super(CDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(depth, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, depth, height, width)
            flat_dim = self.conv(dummy).view(1, -1).size(1)

        # use fc1_dim and fc2_dim for the two-layer MLPs
        self.value = nn.Sequential(
            nn.Linear(flat_dim, fc1_dim), nn.SiLU(),
            nn.Linear(fc1_dim, fc2_dim), nn.SiLU(),
            nn.Linear(fc2_dim, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(flat_dim, fc1_dim), nn.SiLU(),
            nn.Linear(fc1_dim, fc2_dim), nn.SiLU(),
            nn.Linear(fc2_dim, action_space)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move the channels to the start, 0th index is the batch
        x = x.moveaxis(3, 1)
        z = self.conv(x)
        z = z.flatten(1)
        v = self.value(z)
        a = self.advantage(z)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
```
</details>

First I looked up and tried to understand what a neural network is. After I got a pretty good idea about what it is I looked up convolutional neural networks and some examples. Learning them was easier than I expected since I already knew a bit of image processing and was familiar with the terminology and methods used. While searching about activation functions I came accross [this website](https://mbrenndoerfer.com/writing/ffn-activation-functions) that talked about liner unit functions. In my research I already found that ReLU made learning faster in most cases than sigmoid, but thanks to this article I learned there are other ones that get used and decided to use SiLU instead of ReLU for it, because ReLU risks "dead neurons" to stop learning entirely and new architectures prefer using SiLU instead.

The network consists of:
- Convolutional feature extractor: three convolutional blocks produce a hidden convolutional feature map.
- Flattened hidden vector: convolutional output is flattened to a single hidden vector per sample.
- Two fully connected layers from that hidden vector: a value head (produces $V(s)$ ) and an advantage head (produces $A(s,a)$ ).
- With a final output of an estimate of $Q(s,a) = V(s) + A(s,a) − \text{mean}_a A(s,a)$

To train a deep network we need examples for it to generalize the pattern. So we use a replay memory if we are training and append each experience of the agent during training. Than we use them in a batch to train the deep network.
<details>
<summary>Code</summary>

```py
from torch import tensor
from collections import deque
import random
from typing import Deque, Optional, Tuple, NewType, List

Transition = NewType('Transition', Tuple[tensor, tensor, tensor, tensor, bool])

class ReplayMemory():
    """
    A fixed-capacity FIFO replay buffer.
    Type parameter T is the transition type stored
    """

    def __init__(self, max_length: int, seed: Optional[int] = None) -> None:
        self.memory: Deque[Transition] = deque(maxlen=max_length)
        if seed is not None:
            random.seed(seed)

    def append(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, sample_size: int) -> List[Transition]:
        return random.sample(list(self.memory), sample_size)

    def __len__(self) -> int:
        return len(self.memory)
```
</details>

After that I did the epsilon greedy algorithm

<details>
<summary>Code</summary>

```py
                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # argmax finds the index of the largest element.
                        action = policy_net(state.unsqueeze(0)).argmax()

```
</details>

If epsilon is bigger than the random value we do a random action, otherwise we do whatever our policy says. The epsilon starts with a value of 1 and decays over time by getting multiplied by a constant epsilon decay (0.99995), until it reaches a minimum of 0.05 which means it will act randomly only 5% of the times (exploration). So our agent starts of completely random and uses it's learned policy, getting less and less random over time. Since we aren't doing training and just evaluating a step, we disable gradient calculations to save on processing.

I chose the hyperparameters mostly at random based on my talk with the professor and they worked out great on the second try with 2 fully connected 512 node layers, with a learning rate of 0.0001 and discount gamma of 0.99.

We use 2 networks, one policy network that will train and one target network to estimate future Q values. They get initialized identically and initially target network copies all the weights and biases from the policy network to itself. The copy also occurs at a sync stage, that I set as every 10 steps.

At first setting the frame skip to anything higher than 1 felt like it would learn worse but the resulting agent didn't learn like I imagined and it was quite the opposite. The reason was I had to give the agent some time to see the results of its action, so setting frame skip to 4 worked out well. If we look at the graphs we can see the 1 frame skip model performs the worst overall.

<img width="1064" height="772" alt="graph" src="https://github.com/user-attachments/assets/eb9b55ea-63ab-42f2-b551-1d3c4c546172" />


I will reference this graph multiple times throughout the remainder of the document.

## Resulting agent from 1 frame skip (Trained for 29k episodes)

https://github.com/user-attachments/assets/228ff07f-7493-49d3-94a4-70792659fb40


## Agent trained with 4 frame skip (Trained for 20k episodes)

https://github.com/user-attachments/assets/7d4d6464-fafe-43a7-bc15-2650fafee1ae




We can see if we train the agent with 4 frame skip even training it for 20k episodes we get a better result
If we continue training this agent we get a pretty good agent as a result

## Agent trained with 4 frame skip (Trained for 65k episodes)

https://github.com/user-attachments/assets/6efd85aa-6be4-49c0-853c-8a978457f9ca


Perfect, it's done right? That's what I tought but while I was messing with less trained agents I found that the agent does this

## Agent trained for 10k episodes

https://github.com/user-attachments/assets/914ee62c-78af-46f2-bd45-534c6ecc6b22


It prefers waiting rather than moving or shooting. Since it gets the same reward no matter if it's moving or not and get's less punishment if it doesn't shoot (without killing the enemy) waiting seems as good as moving. To fix this I made it so that if it doesn't shoot in an episode at all, right before termination it get's a negative reward 1 lower than the punishment for shooting a bullet to encourage it to shoot more rather than waiting and doing nothing at all.
<details>
 <summary>Code in gymnasium_env/envs/vizdoom_gymnasium.py VizdoomGymnasiumEnv:#L166</summary>

```py
        # If no ammo is used punish, attempt to prevent trashing
        if terminated and (info.get("ammo") >= 50):
            reward -= 6
```
</details>

So our current reward function is as follows:

$$
R = 
\begin{cases} 
106 & \text{if monster is killed} \\
-5 \cdot n_{\text{shots}} & \text{for every shot taken} \\
-1 & \text{for each time step (living reward)} \\
-6 & \text{if } n_{\text{shots}} = 0 \text{ at the end of the episode}
\end{cases}
$$

And this worked out really well, I got pretty good results for training it for only 6k episodes with a lower epsilon decay (0.995) to make it decay faster
## Agent trained with new Reward function for 6k episodes

https://github.com/user-attachments/assets/8bba7b33-b57f-4f0c-8ec1-812125223e88


Than I upped the epsilon decay back to what it was and trained it
## Agent trained with new reward function for 10k episodes

https://github.com/user-attachments/assets/e4527b65-b065-4e5a-948d-0705ce86c0d7


## Agent trained with new reward function for 20k episodes

https://github.com/user-attachments/assets/3aaeab49-0a63-4404-83bc-3e350ec3c810


## Agent trained with new reward function for 65k episodes

https://github.com/user-attachments/assets/cee24de4-b851-408a-8b06-971d4b32c592
