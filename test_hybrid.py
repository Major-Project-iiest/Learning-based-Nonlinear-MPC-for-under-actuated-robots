import numpy as np
import matplotlib.pyplot as plt

from dynamics_learner.gp import SlidingWindowGP
from dynamics_learner.nn import ResidualNN

# ---------- Hybrid wrapper ----------
class HybridResidual:
    def __init__(self, x_dim, u_dim, r_dim, alpha=0.5, gp_kwargs=None, nn_kwargs=None):
        self.alpha = float(alpha)
        self.gp = SlidingWindowGP(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim, **(gp_kwargs or {}))
        self.nn = ResidualNN(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim, **(nn_kwargs or {}))

    def update(self, batch):
        self.gp.update(batch)
        self.nn.update(batch)

    def predict(self, x, u):
        mg, vg = self.gp.predict(x, u)
        mn, vn = self.nn.predict(x, u)
        mean = self.alpha * mg + (1.0 - self.alpha) * mn
        var  = self.alpha * vg + (1.0 - self.alpha) * vn
        return mean, var


# ---------- Fake residual generator ----------
def fake_residual(x, u):
    return 0.5 * np.asarray(x) + 0.1 * np.asarray(u) + 0.01 * np.random.randn(len(x))


# ---------- Rolling metric ----------
def rolling_metric(y_true, y_pred, window=20, metric="mse"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    T, D = y_true.shape
    out = np.zeros(T)
    for t in range(T):
        s = max(0, t - window + 1)
        yt = y_true[s:t+1]
        yp = y_pred[s:t+1]
        if metric == "mse":
            err = np.mean((yp - yt) ** 2)
        else:
            err = np.mean(np.abs(yp - yt))
        out[t] = err
    return out


# ---------- Main experiment ----------
def main():
    rng = np.random.default_rng(0)
    x_dim, u_dim, r_dim = 4, 1, 4
    T = 400
    train_every = 32
    roll_window = 25
    compare_dim = 0

    gp = SlidingWindowGP(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim, maxlen=200)
    nn = ResidualNN(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim)
    hybrid = HybridResidual(x_dim=x_dim, u_dim=u_dim, r_dim=r_dim, alpha=0.5)

    truths, gp_means, nn_means, hy_means = [], [], [], []
    batch = []

    for t in range(T):
        x = rng.standard_normal(x_dim)
        u = rng.standard_normal(u_dim)
        r = fake_residual(x, u)
        batch.append((x, u, r))

        if len(batch) >= train_every:
            gp.update(batch)
            nn.update(batch)
            hybrid.update(batch)
            batch = []

        mg, _ = gp.predict(x, u)
        mn, _ = nn.predict(x, u)
        mh, _ = hybrid.predict(x, u)

        truths.append(r)
        gp_means.append(mg)
        nn_means.append(mn)
        hy_means.append(mh)

    truths, gp_means, nn_means, hy_means = map(np.array, (truths, gp_means, nn_means, hy_means))

    # Rolling metrics
    gp_mse = rolling_metric(truths, gp_means, roll_window, "mse")
    nn_mse = rolling_metric(truths, nn_means, roll_window, "mse")
    hy_mse = rolling_metric(truths, hy_means, roll_window, "mse")

    gp_mae = rolling_metric(truths, gp_means, roll_window, "mae")
    nn_mae = rolling_metric(truths, nn_means, roll_window, "mae")
    hy_mae = rolling_metric(truths, hy_means, roll_window, "mae")

    # Summary numbers
    def summary(name, pred):
        mse = np.mean((pred - truths) ** 2)
        mae = np.mean(np.abs(pred - truths))
        print(f"{name}: MSE={mse:.6f}, MAE={mae:.6f}")

    print("==== Overall Performance ====")
    summary("GP", gp_means)
    summary("NN", nn_means)
    summary("Hybrid", hy_means)

    # ---------- Plots ----------
    t = np.arange(T)

    plt.figure(figsize=(11, 4))
    plt.plot(t, truths[:, compare_dim], label="True residual", color="black")
    plt.plot(t, gp_means[:, compare_dim], label="GP")
    plt.plot(t, nn_means[:, compare_dim], label="NN")
    plt.plot(t, hy_means[:, compare_dim], label="Hybrid", linestyle="--")
    plt.title(f"Residual tracking (dim {compare_dim})")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 4))
    plt.plot(t, gp_mse, label="GP")
    plt.plot(t, nn_mse, label="NN")
    plt.plot(t, hy_mse, label="Hybrid")
    plt.title("Rolling MSE")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 4))
    plt.plot(t, gp_mae, label="GP")
    plt.plot(t, nn_mae, label="NN")
    plt.plot(t, hy_mae, label="Hybrid")
    plt.title("Rolling MAE")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
