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
            nn.SiLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, depth, height, width)
            flat_dim = self.conv(dummy).view(1, -1).size(1)

        # use fc1_dim and fc2_dim for the two-layer MLPs
        self.value = nn.Sequential(
            nn.Linear(flat_dim, fc1_dim),
            nn.SiLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.SiLU(),
            nn.Linear(fc2_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(flat_dim, fc1_dim),
            nn.SiLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.SiLU(),
            nn.Linear(fc2_dim, action_space),
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
