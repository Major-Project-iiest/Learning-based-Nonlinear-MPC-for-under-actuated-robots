# dynamics_learner/window.py
import collections
import numpy as np
from typing import List, Tuple, Sequence

class SlidingWindow:
    """Simple FIFO sliding window storing (x, u, residual)."""

    def __init__(self, maxlen: int = 1000):
        self.buffer = collections.deque(maxlen=maxlen)

    def add(self, x: Sequence[float], u: Sequence[float], residual: Sequence[float]) -> None:
        self.buffer.append((np.asarray(x, dtype=float).copy(),
                            np.asarray(u, dtype=float).copy(),
                            np.asarray(residual, dtype=float).copy()))

    def sample(self, batch_size: int = 64) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        n = len(self.buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)
        idxs = np.random.choice(n, size=batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

    def all(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)
