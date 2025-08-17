# dynamics_learner/gp.py
import numpy as np
from typing import Sequence, Tuple, Optional, List
from .base import DynamicsLearner

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float, variance: float):
    """
    RBF kernel (squared exponential).
    X1: [n1, d], X2: [n2, d]
    returns: [n1, n2]
    """
    # squared distance
    X1s = np.sum(X1**2, axis=1, keepdims=True)
    X2s = np.sum(X2**2, axis=1, keepdims=True)
    d2 = X1s - 2*X1.dot(X2.T) + X2s.T
    return variance * np.exp(-0.5 * d2 / (lengthscale**2))

class SlidingWindowGP(DynamicsLearner):
    """
    Exact GP on a sliding-window dataset.
    - stores up to maxlen samples
    - supports multi-output Y (shape [N, r_dim])
    - prediction: mean (r_dim,) and var (r_dim,)
    Note: exact GP has O(N^3) training cost where N = window size. Keep maxlen small (~200-1000).
    """

    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 r_dim: int,
                 maxlen: int = 500,
                 lengthscale: float = 1.0,
                 variance: float = 1.0,
                 noise_variance: float = 1e-3,
                 jitter: float = 1e-8):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.in_dim = x_dim + u_dim
        self.r_dim = r_dim
        self.maxlen = maxlen
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_variance = noise_variance
        self.jitter = jitter

        # Buffers
        self.X = np.zeros((0, self.in_dim), dtype=float)
        self.Y = np.zeros((0, self.r_dim), dtype=float)

        # Precomputed matrices
        self.K = None      # kernel matrix (N,N)
        self.L = None      # cholesky of K + noise
        self.alpha = None  # solver for predictions: solve(K+σI, Y)

    def _form_input(self, x: Sequence[float], u: Sequence[float]) -> np.ndarray:
        return np.asarray(np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]), dtype=float).reshape(1, -1)

    def predict(self, x: Sequence[float], u: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.X) == 0:
            # No data => zero mean, large variance
            return np.zeros(self.r_dim), np.full(self.r_dim, self.variance + self.noise_variance)

        x_star = self._form_input(x, u)  # [1, in_dim]
        K_star = rbf_kernel(x_star, self.X, self.lengthscale, self.variance)  # [1, N]
        K_ss = rbf_kernel(x_star, x_star, self.lengthscale, self.variance).reshape(())  # scalar

        # Solve for mean: k_star @ K^{-1} @ Y
        # using precomputed cholesky L and alpha (= K^{-1}Y)
        try:
            # alpha shape [N, r_dim]
            alpha = self.alpha  # precomputed
            mean = (K_star @ alpha).ravel()  # shape (r_dim,)
            # predictive variance: k_ss - k_star K^{-1} k_star^T
            # compute v = solve(L, k_star^T)
            v = np.linalg.solve(self.L, K_star.T)  # [N,1]
            var_scalar = K_ss - (v.T @ v).ravel()[0]
            var_scalar = max(var_scalar, 1e-12)
            var = np.full(self.r_dim, var_scalar) + self.noise_variance
            return mean, var
        except Exception:
            # fallback: compute directly (slow)
            K = self.K + (self.noise_variance + self.jitter) * np.eye(len(self.X))
            try:
                L = np.linalg.cholesky(K)
                invK = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
                mean = (K_star @ invK @ self.Y).ravel()
                var_scalar = K_ss - (K_star @ invK @ K_star.T).ravel()[0]
                var = np.full(self.r_dim, max(var_scalar, 1e-12) + self.noise_variance)
                return mean, var
            except np.linalg.LinAlgError:
                # extremely fallback
                return (K_star @ np.zeros((self.X.shape[0], self.r_dim))).ravel(), np.full(self.r_dim, self.variance)

    def update(self, batch: Sequence[Tuple[Sequence[float], Sequence[float], Sequence[float]]]) -> None:
        """
        Append data from batch into the sliding window and recompute factorization.
        Keep at most maxlen samples.
        """
        if len(batch) == 0:
            return
        X_new = []
        Y_new = []
        for x,u,r in batch:
            X_new.append(np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]))
            Y_new.append(np.asarray(r).ravel())
        X_new = np.asarray(X_new, dtype=float)
        Y_new = np.asarray(Y_new, dtype=float)

        if self.X.shape[0] == 0:
            self.X = X_new.copy()
            self.Y = Y_new.copy()
        else:
            self.X = np.vstack([self.X, X_new])
            self.Y = np.vstack([self.Y, Y_new])

        # Truncate to most recent samples
        if len(self.X) > self.maxlen:
            self.X = self.X[-self.maxlen:]
            self.Y = self.Y[-self.maxlen:]

        # Recompute kernel and cholesky
        K = rbf_kernel(self.X, self.X, self.lengthscale, self.variance)
        K += (self.noise_variance + self.jitter) * np.eye(len(K))
        try:
            L = np.linalg.cholesky(K)
            # compute alpha = K^{-1} Y using cholesky solves
            # Solve L v = Y  => v = solve(L, Y), then alpha = solve(L.T, v)
            v = np.linalg.solve(L, self.Y)
            alpha = np.linalg.solve(L.T, v)  # shape [N, r_dim]
            self.K = K
            self.L = L
            self.alpha = alpha
        except np.linalg.LinAlgError:
            # Numerical issue: jitter more and fallback
            jitter2 = 1e-6
            K += jitter2 * np.eye(K.shape[0])
            L = np.linalg.cholesky(K)
            v = np.linalg.solve(L, self.Y)
            alpha = np.linalg.solve(L.T, v)
            self.K = K
            self.L = L
            self.alpha = alpha

    def set_mode(self, mode: str = 'eval') -> None:
        # GP doesn't need train/eval modes but keep API compatible
        return
