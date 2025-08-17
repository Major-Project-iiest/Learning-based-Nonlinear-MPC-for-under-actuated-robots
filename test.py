# test_learners.py
import numpy as np
import matplotlib.pyplot as plt
from dynamics_learner.gp import SlidingWindowGP
from dynamics_learner.nn import ResidualNN

# ------------------ Fake residual dynamics ------------------
def fake_residual(x, u):
    # residual = linear combo + small noise
    return np.asarray(x) * 0.5 + np.asarray(u) * 0.1 + 0.01*np.random.randn(len(x))

# ------------------ Test GP ------------------
gp = SlidingWindowGP(x_dim=4, u_dim=1, r_dim=4, maxlen=50)

gp_means, gp_vars, gp_truths = [], [], []
x = np.zeros(4)
for t in range(30):
    u = np.random.randn(1)
    r = fake_residual(x, u)
    gp.update([(x, u, r)])    # update with single sample
    mean, var = gp.predict(x, u)
    gp_means.append(mean.copy())
    gp_vars.append(var.copy())
    gp_truths.append(r.copy())

gp_means = np.array(gp_means)
gp_vars = np.array(gp_vars)
gp_truths = np.array(gp_truths)

# ------------------ Test NN ------------------
nn = ResidualNN(x_dim=4, u_dim=1, r_dim=4)

nn_means, nn_vars, nn_truths = [], [], []
batch = []
for t in range(100):
    x = np.random.randn(4)
    u = np.random.randn(1)
    r = fake_residual(x, u)
    batch.append((x, u, r))
    if len(batch) >= 32:   # train every 32 samples
        nn.update(batch)
        batch = []
    mean, var = nn.predict(x, u)
    nn_means.append(mean.copy())
    nn_vars.append(var.copy())
    nn_truths.append(r.copy())

nn_means = np.array(nn_means)
nn_vars = np.array(nn_vars)
nn_truths = np.array(nn_truths)

# ------------------ Plotting ------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# GP results
axs[0].plot(gp_truths[:, 0], label="True residual (dim 0)", color="black")
axs[0].plot(gp_means[:, 0], label="GP mean", color="blue")
axs[0].fill_between(range(len(gp_vars)), 
                    gp_means[:, 0] - np.sqrt(gp_vars[:, 0]),
                    gp_means[:, 0] + np.sqrt(gp_vars[:, 0]),
                    color="blue", alpha=0.2, label="GP ±1 std")
axs[0].set_title("Sliding Window GP (dim 0)")
axs[0].legend()

# NN results
axs[1].plot(nn_truths[:, 0], label="True residual (dim 0)", color="black")
axs[1].plot(nn_means[:, 0], label="NN mean", color="red")
axs[1].fill_between(range(len(nn_vars)), 
                    nn_means[:, 0] - np.sqrt(nn_vars[:, 0]),
                    nn_means[:, 0] + np.sqrt(nn_vars[:, 0]),
                    color="red", alpha=0.2, label="NN ±1 std")
axs[1].set_title("Residual NN with MC Dropout (dim 0)")
axs[1].legend()

plt.tight_layout()
plt.show() 