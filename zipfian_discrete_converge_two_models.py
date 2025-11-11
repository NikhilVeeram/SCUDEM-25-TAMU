import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Core simulation parameters
STEPS = 1000                          # number of iterations to run the convergence process
MODEL_IMPROVEMENT_RATE = 0.02          # K_M: how fast models move toward current dataset
SELF_LEARNING = 0.02                   # k_S: self-learning rate when S(t) is "on"
DATA_IMPROVEMENT_RATE = 0.05           # K_D: how fast datasets move toward public 2025 dataset
PLOT_INTERVAL = 5
START_SELF_LEARNING_LATE = True
SELF_LEARNING_START_FRACTION = 0.25    # t_start / T_total in S(t)
EXPERIMENT_NAME = "Zipf Two Models With Late Self Training"

# Zipf and vocabulary assumptions
VOCAB_SIZE = 400                       # number of token ranks
ZIPF_S = 1.1                           # Zipf exponent; ~1.0–1.1 is language-like

# Dataset vs model scale and human content injection
DATASET_SIZE_RATIO = 100.0            # D_i(t) ~ 100x larger than f_i(t)
HUMAN_CONTENT_RATE = 0.02             # k_h in h(t) = k_h * t (scaled by distribution)
TAIL_FRACTION = 0.3                   # fraction of ranks treated as "tail" for q metrics

# Rank axis: i = 1 (most frequent token) ... VOCAB_SIZE (rarest)
ranks = np.arange(1, VOCAB_SIZE + 1)

# Base Zipf prior over ranks
zipf_base = 1.0 / np.power(ranks, ZIPF_S)
zipf_base = zipf_base / zipf_base.sum()

def make_zipf_gaussian(center_rank, width, base=None):
    """
    Construct a Zipfian distribution modulated by a Gaussian envelope
    over ranks. This represents 'Zipfian language' with a topical or
    stylistic focus around a particular rank band.
    """
    if base is None:
        base = zipf_base
    g = np.exp(-0.5 * ((ranks - center_rank) / width) ** 2)
    dist = base * g
    dist = np.clip(dist, 1e-12, None)
    dist /= dist.sum()
    return dist

# Initial datasets and model outputs (all Zipfian)
# Think of y_target/y_target_2 as 2023 datasets for each model,
# y_second_target as the 2025 public dataset, and human_dist as new human content.
# --------------------------------------------------------------------------
# NEW: Zipfian + perturbation initialization for datasets and model outputs
# --------------------------------------------------------------------------
def make_zipf_variant(center_rank_shift=0.0, slope_delta=0.0, noise_scale=0.02):
    local_s = ZIPF_S + slope_delta
    # Avoid zero or negative bases
    shifted = np.clip(ranks + center_rank_shift, 1, None)
    base = 1.0 / np.power(shifted, local_s)
    noise = np.random.normal(0.0, noise_scale, size=base.shape)
    noisy = base * (1.0 + noise)
    noisy = np.clip(noisy, 1e-12, None)
    noisy /= noisy.sum()
    return noisy


# Use quasi-Zipfian variants for everything
y_target   = make_zipf_variant(center_rank_shift=0,  slope_delta=0.02,  noise_scale=0.01)   # dataset for model 1 (2023)
y_start    = make_zipf_variant(center_rank_shift=5,  slope_delta=-0.03, noise_scale=0.015)  # model 1 initial output
y_target_2 = make_zipf_variant(center_rank_shift=-5, slope_delta=0.03,  noise_scale=0.01)   # dataset for model 2 (2023)
y_start_2  = make_zipf_variant(center_rank_shift=10, slope_delta=-0.02, noise_scale=0.02)   # model 2 initial output

# Public dataset and human data remain pure Zipf-Gaussian blends (2025 composition)
y_second_target = make_zipf_gaussian(center_rank=90, width=35)
human_dist      = make_zipf_gaussian(center_rank=220, width=80)
# --------------------------------------------------------------------------


# Baselines to measure drift and collapse against
y_target_og = y_target.copy()             # original 2023 dataset for model 1
y_second_target_og = y_second_target.copy()  # original 2025 public dataset

eps = 1e-12
entropy_baseline = -np.sum(y_second_target_og * np.log(y_second_target_og + eps))
tail_start = int((1.0 - TAIL_FRACTION) * VOCAB_SIZE)
baseline_tail_mass = np.sum(y_second_target_og[tail_start:])

# Time series diagnostics
mse = []
q_stagnation = []     # similarity to 2023 dataset
q_tail_ratio = []     # tail mass relative to 2025 baseline
q_entropy_ratio = []  # entropy relative to 2025 baseline

# Plot initial Zipfian targets and starts
plt.figure(figsize=(8, 5))
plt.plot(ranks, y_target, color="orange", label="target 1 (2023)")
plt.plot(ranks, y_start, color="blue", label="start 1")
plt.plot(ranks, y_target_2, color="red", label="target 2 (2023)")
plt.plot(ranks, y_start_2, color="purple", label="start 2")
plt.plot(ranks, y_second_target, color="black", label="public dataset 2025")
plt.xlabel("Token rank (1 = most frequent)")
plt.ylabel("Probability")
plt.title("Initial Zipfian Target and Start Distributions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Store snapshots of model 1 for convergence plot
snapshots_model1 = []

for step in range(STEPS):
    t_frac = step / STEPS

    # S(t): onset of self-learning
    if START_SELF_LEARNING_LATE and t_frac < SELF_LEARNING_START_FRACTION:
        self_learning_rate = 0.0
    else:
        self_learning_rate = SELF_LEARNING

    # h(t) = k_h * t, applied via a human Zipfian distribution
    h_t = HUMAN_CONTENT_RATE * t_frac

    # Model updates: df_i/dt = (K_M + S_B(t)) (D_i - f_i)
    y_start = (1 - MODEL_IMPROVEMENT_RATE - self_learning_rate) * y_start + \
              (MODEL_IMPROVEMENT_RATE + self_learning_rate) * y_target
    y_start = np.clip(y_start, 1e-12, None)
    y_start /= y_start.sum()

    y_start_2 = (1 - MODEL_IMPROVEMENT_RATE - self_learning_rate) * y_start_2 + \
                (MODEL_IMPROVEMENT_RATE + self_learning_rate) * y_target_2
    y_start_2 = np.clip(y_start_2, 1e-12, None)
    y_start_2 /= y_start_2.sum()

    # Dataset updates: dD_i/dt = h(t) + K_D D_i(t) + (weak) self-learning contamination
    contam_scale = self_learning_rate / DATASET_SIZE_RATIO

    self1 = y_start * y_start
    self1 /= self1.sum()
    self2 = y_start_2 * y_start_2
    self2 /= self2.sum()

    # Datasets for each model drift toward the public dataset and get a bit contaminated
    y_target = (1 - DATA_IMPROVEMENT_RATE - contam_scale) * y_target + \
               DATA_IMPROVEMENT_RATE * y_second_target + \
               contam_scale * self1
    y_target = np.clip(y_target, 1e-12, None)
    y_target /= y_target.sum()

    y_target_2 = (1 - DATA_IMPROVEMENT_RATE - contam_scale) * y_target_2 + \
                 DATA_IMPROVEMENT_RATE * y_second_target + \
                 contam_scale * self2
    y_target_2 = np.clip(y_target_2, 1e-12, None)
    y_target_2 /= y_target_2.sum()

    # Public dataset combines its own previous state, original 2025 data, synthetic AI outputs,
    # and time-growing human content h(t)
    human_injection = h_t * human_dist
    y_second_target = (1 - DATA_IMPROVEMENT_RATE - 2 * contam_scale) * y_second_target + \
                      DATA_IMPROVEMENT_RATE * y_second_target_og + \
                      contam_scale * (self1 + self2) + \
                      human_injection
    y_second_target = np.clip(y_second_target, 1e-12, None)
    y_second_target /= y_second_target.sum()

    # Quality and collapse diagnostics for model 1
    cosine_stag = np.dot(y_start, y_target_og) / (np.linalg.norm(y_start) * np.linalg.norm(y_target_og))
    q_stagnation.append(cosine_stag)

    entropy_curr = -np.sum(y_start * np.log(y_start + eps))
    q_entropy_ratio.append(entropy_curr / (entropy_baseline + eps))

    tail_mass_curr = np.sum(y_start[tail_start:])
    q_tail_ratio.append(np.log10((tail_mass_curr + eps) / (baseline_tail_mass + eps)))

    # MSE between model 1 and the 2025 public dataset (captures collapse vs adaptation)
    mse.append(np.mean((y_start - y_second_target_og) ** 2))

    # Save snapshots for plotting
    if (step % PLOT_INTERVAL) == 0 or step == STEPS - 1:
        snapshots_model1.append(y_start.copy())

# Convergence plot for model 1
plt.figure(figsize=(8, 5))
num = len(snapshots_model1)
for idx, dist in enumerate(snapshots_model1):
    alpha = 0.1 + 0.9 * (idx / max(num - 1, 1))
    plt.plot(ranks, dist, color="blue", alpha=alpha)
plt.plot(ranks, y_target_og, "r--", label="original 2023 dataset (model 1)")
plt.plot(ranks, y_second_target_og, "k--", label="public 2025 dataset")
plt.xlabel("Token rank (1 = most frequent)")
plt.ylabel("Probability")
plt.title("Convergence of Model 1 under Zipfian Dynamics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# MSE curve over time

print("Length of MSE list:", len(mse))
print("First few MSE values:", mse[:5])
print("Last few MSE values:", mse[-5:])

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(mse)), mse, color="red")   # Explicit x-axis = iteration numbers
plt.xlabel("Iteration")
plt.ylabel("MSE (model 1 vs public 2025 dataset)")
plt.title("MSE between Start and Public Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()


# q(t) diagnostics
plt.figure(figsize=(8, 5))
plt.plot(q_stagnation, label="q_stagnation (similarity to 2023)")
plt.plot(q_tail_ratio, label="q_tail_ratio (tail mass ratio)")
plt.plot(q_entropy_ratio, label="q_entropy_ratio (entropy ratio)")
plt.xlabel("Iteration")
plt.ylabel("Quality metrics")
plt.title("Stagnation and Diversity Diagnostics over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save MSE in the familiar format for cross-experiment comparison
csv_path = "mse.csv"
col_name = EXPERIMENT_NAME
df_new = pd.DataFrame({col_name: mse})

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    max_len = max(len(df), len(df_new))
    df = df.reindex(range(max_len))
    df[col_name] = pd.Series(mse)
else:
    df = df_new

df.to_csv(csv_path, index=False)

# Save quality metrics for deeper analysis
quality_df = pd.DataFrame({
    "q_stagnation": q_stagnation,
    "q_tail_ratio": q_tail_ratio,
    "q_entropy_ratio": q_entropy_ratio
})
quality_df.to_csv("quality_metrics.csv", index=False)

# Aggregate MSE plot from mse.csv (all experiments)
if os.path.exists(csv_path):
    df_plot = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    for col in df_plot.columns:
        series = df_plot[col]
        mask = series.notna()
        plt.plot(np.arange(len(series[mask])), series[mask], label=col)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("MSEs from CSV (all experiments)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
