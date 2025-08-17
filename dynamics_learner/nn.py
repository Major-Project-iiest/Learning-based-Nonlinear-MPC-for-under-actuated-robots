import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Sequence, Tuple, Optional
from .base import DynamicsLearner


# 🔹 Simple MLP with Dropout
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: Sequence[int] = (64, 64, 32),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualNN(DynamicsLearner):
    """
    Neural-network residual learner with MC-dropout.
    - Uses MSE loss.
    - predict() returns (mean, var).
    """

    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 r_dim: int,
                 hidden: Sequence[int] = (64, 64),
                 lr: float = 1e-3,
                 dropout: float = 0.3,
                 device: Optional[str] = None):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.in_dim = x_dim + u_dim
        self.device = torch.device(
            device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Model + optimizer
        self.model = MLP(self.in_dim, r_dim, hidden=hidden, dropout=dropout).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Configs
        self._mc_passes_default = 20

    def predict(self, x: Sequence[float], u: Sequence[float],
                mc_passes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict residual and uncertainty using MC-dropout.
        """
        mc_passes = mc_passes or self._mc_passes_default

        # Force dropout active
        self.model.train()

        inp = np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]).astype(np.float32)
        xt = torch.from_numpy(inp).unsqueeze(0).to(self.device)

        preds = []
        with torch.no_grad():
            for _ in range(mc_passes):
                out = self.model(xt)
                preds.append(out.cpu().numpy().ravel())
        preds = np.stack(preds, axis=0)

        mean = preds.mean(axis=0)
        var = preds.var(axis=0) + 1e-8
        return mean, var

    def update(self,
               batch: Sequence[Tuple[Sequence[float], Sequence[float], Sequence[float]]],
               epochs: int = 25,
               batch_size: int = 32,
               patience: int = 5) -> None:
        """
        Update on given batch with early stopping.
        """
        if len(batch) == 0:
            return

        # Dataset
        X, Y = [], []
        for x, u, r in batch:
            X.append(np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]))
            Y.append(np.asarray(r).ravel())
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        # Training loop with early stopping
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = self.criterion(pred, yb)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self.model.eval()
