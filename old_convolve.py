import numpy as np
import matplotlib.pyplot as plt

target_func = lambda x, mu=0.0, sigma=1.0: (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
start_func = lambda x: 0.25 * np.ones_like(x)
second_target = lambda x: 0.5 * np.exp(-0.5 * (x / 0.5) ** 2)

# set up grid and kernel
x = np.linspace(-10, 10, 2001)
dx = x[1] - x[0]
kernel = target_func(x)             # use provided target as convolution kernel
kernel /= kernel.sum() * dx         # normalize to unit area

# initial function (from provided start_func) and normalize to unit area
f = start_func(x)
f /= (f.sum() * dx)

# perform repeated operation to move `f` toward a target
# two modes are supported:
#  - 'conv': repeated convolution with `kernel` (original behavior)
#  - 'mix' : convex mixing f <- (1-alpha)*f + alpha*target (slowly approaches target)
mode = 'mix'        # choose 'mix' or 'conv'
alpha = 0.05       # mixing rate used only for mode='mix'
n_steps = 100
snapshots = [f.copy()]

if mode == 'conv':
    # repeated convolution (original script behavior)
    F_k = np.fft.rfft(kernel)           # precompute kernel FFT
    for i in range(n_steps):
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F_k, n=f.size)
        f = np.maximum(conv, 0.0)       # remove tiny negative roundoff
        f /= (f.sum() * dx)             # re-normalize to unit area
        snapshots.append(f.copy())
else:
    # mixing mode: each step moves f slightly toward the `target_func`
    target = target_func(x)
    target /= (target.sum() * dx)
    for i in range(n_steps):
        f = (1.0 - alpha) * f + alpha * target
        f = np.maximum(f, 0.0)
        f /= (f.sum() * dx)
        snapshots.append(f.copy())

# choose a few steps to plot (including start and final)
plot_indices = [0, 1, 2, 5, 10, n_steps]
plot_indices = [i for i in plot_indices if i < len(snapshots)]

plt.figure(figsize=(8, 5))
plt.plot(x, target_func(x), 'k--', label='target')
colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
for idx, c in zip(plot_indices, colors):
    plt.plot(x, snapshots[idx], color=c, label=f'step {idx}')
plt.legend()
plt.xlabel('x')
plt.ylabel('density')
plt.title(f'Progression (mode={mode})')
plt.show()