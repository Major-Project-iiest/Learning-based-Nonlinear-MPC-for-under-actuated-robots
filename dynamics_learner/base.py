# dynamics_learner/base.py
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Any
import numpy as np

class DynamicsLearner(ABC):
    """Facade for dynamics learners."""

    @abstractmethod
    def predict(self, x: Sequence[float], u: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict residual mean and variance for input (x,u).
        Returns:
            mean: np.array(shape=(r_dim,))
            var:  np.array(shape=(r_dim,)) or scalar per-dimension
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Sequence[Tuple[Sequence[float], Sequence[float], Sequence[float]]]) -> None:
        """
        Update model with a batch of (x, u, residual) samples.
        batch: sequence of tuples: (x, u, residual)
        """
        raise NotImplementedError

    def set_mode(self, mode: str = 'eval') -> None:
        """
        Optional: 'train' or 'eval' (e.g. to enable dropout).
        Default is eval.
        """
        pass
