import numpy as np
import matplotlib.pyplot as plt

target_func = lambda x, mu=0.0, sigma=1.0: 3 * (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
start_func = lambda x: 0.25 * np.ones_like(x)
# second_target = lambda x: 0.5*np.exp(-10*(x+2))
second_target = lambda x, mu=0.0, sigma=1.0: (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((x+1) - mu) / sigma) ** 2)

def taylor_coeffs(func, n, a=0.0, radius=0.5):
    """
    Return Taylor coefficients up to degree n (inclusive) of func around point a.
    This implementation fits a polynomial of degree n to function samples
    taken at Chebyshev nodes in [a-radius, a+radius] using a stable least-squares
    solve of the Vandermonde system, and returns the monomial coefficients
    c_k so that f(x) ≈ sum_k c_k (x-a)^k (hence c_k = f^(k)(a)/k!).
    The Chebyshev sampling avoids catastrophic cancellation and the explicit
    h**k division used by forward finite differences.
    """
    m = n + 1
    if radius <= 0:
        radius = 0.5

    # Chebyshev nodes in [-radius, radius], mapped around a
    i = np.arange(m)
    x_nodes = a + radius * np.cos((2 * i + 1) * np.pi / (2 * m))

    # Ensure we evaluate the function for each node even if func is not vectorized
    y = np.array([func(xi) for xi in x_nodes], dtype=float)

    # Build Vandermonde matrix with columns (x-a)^k for k=0..n (increasing order)
    V = np.vander(x_nodes - a, N=m, increasing=True)

    # Solve for coefficients in least-squares sense to mitigate conditioning issues
    coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)

    return coeffs.tolist()

def poly_eval(coeffs, x, a=0.0):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y += c * (x - a) ** k
    return y

a = 0.0
x = np.linspace(-5, 5, 400)

plt.figure(figsize=(8, 5))
plt.plot(x, target_func(x), color="k", label="target func")
plt.plot(x, start_func(x), color="k", label="start func")
plt.plot(x, second_target(x), color="k", label="second target")

degree = 12
coeffs_t = taylor_coeffs(target_func, degree, a=a)
plt.plot(x, poly_eval(coeffs_t, x, a=a), color="blue", linestyle="--", label=f"target Taylor n={degree}")

coeffs_s = taylor_coeffs(start_func, degree, a=a)
plt.plot(x, poly_eval(coeffs_s, x, a=a), color="green", linestyle="--", label=f"start Taylor n={degree}")

coeffs_t2 = taylor_coeffs(second_target, degree, a=a)
plt.plot(x, poly_eval(coeffs_t2, x, a=a), color="orange", linestyle="--", label=f"second target Taylor n={degree}")

plt.axvline(a, color="gray", linewidth=0.5)
plt.legend(loc="best", fontsize="small")
plt.title(f"Taylor approximations around a={a}")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(0, 1)
plt.show()

plot_interval = 50
# interpolate coefficients from start -> target and save checkpoints
steps = 1000
checkpoint_interval = max(1, plot_interval)  # use plot_interval from above to decide how often to save
cs = np.asarray(coeffs_s, dtype=float)
ct = np.asarray(coeffs_t, dtype=float)
ct2 = np.asarray(coeffs_t2, dtype=float)

saved_coeffs = []
saved_coeffs2 = []  # also save second target interpolation for reference
for i, w in enumerate(np.linspace(0.0, 1.0, steps + 1)):
    ci2 = (1.0 - w) * ct + w * ct2
    ci = (1.0 - w) * cs + w * ci2
    if (i % checkpoint_interval) == 0 or i == steps:
        saved_coeffs.append(ci.copy())
        saved_coeffs2.append(ci2.copy())  # also save second target interpolation for reference

# w is just the rate from start -> target
# plot checkpoints with opacity increasing for more recent checkpoints
plt.figure(figsize=(8, 5))
num = len(saved_coeffs)
for idx, ci in enumerate(saved_coeffs):
    alpha = 0.1 + 0.9 * (idx / (num - 1)) if num > 1 else 1.0  # older -> more transparent
    y = poly_eval(ci, x, a=a)
    plt.plot(x, y, color="red", linewidth=1, alpha=alpha)

# highlight final result
final_coeffs = saved_coeffs[-1]
plt.plot(x, poly_eval(final_coeffs, x, a=a), color="red", linewidth=2, label="interpolated final")

# for idx, ci2 in enumerate(saved_coeffs2):
#     alpha = 0.1 + 0.9 * (idx / (num - 1)) if num > 1 else 1.0  # older -> more transparent
#     y = poly_eval(ci2, x, a=a)
#     plt.plot(x, y, color="blue", linewidth=1, alpha=alpha)

# highlight final result
# final_coeffs2 = saved_coeffs2[-1]
# plt.plot(x, poly_eval(final_coeffs2, x, a=a), color="orange", linewidth=2, label="interpolated final 2")

# optional: replot start and target for reference (lower alpha so checkpoints stand out)
plt.plot(x, target_func(x), color="k", linewidth=1, alpha=0.6, label="target func")
plt.plot(x, start_func(x), color="k", linewidth=1, alpha=0.4, label="start func")
plt.plot(x, second_target(x), color="k", linewidth=1, alpha=0.4, label="second target")

plt.axvline(a, color="gray", linewidth=0.5)
plt.legend(loc="best", fontsize="small")
plt.title("Coefficient interpolation checkpoints (older are more transparent)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(0, 1)
plt.show()

