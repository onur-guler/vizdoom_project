from torch import Tensor
from collections import deque
import random
from typing import Deque, Optional, Tuple, TypeAlias, List

Transition: TypeAlias = Tuple[Tensor, Tensor, Tensor, Tensor, bool]


class ReplayMemory:
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
