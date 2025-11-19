import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Core simulation parameters
# ============================
STEPS = 1000
MODEL_IMPROVEMENT_RATE = 0.02
SELF_LEARNING = 0.02
SELF_LEARNING_RAMP_STEPS = 50
DATA_IMPROVEMENT_RATE = 0.05
PLOT_INTERVAL = 5
START_SELF_LEARNING_LATE = True
SELF_LEARNING_START_FRACTION = 0.25
N_MODELS = 10  # <<<<<<<<<<<<<< Ten models

# ============================
# Zipf & vocab
# ============================
VOCAB_SIZE = 400
ZIPF_S = 1.1
ranks = np.arange(1, VOCAB_SIZE + 1)

DATASET_SIZE_RATIO = 100.0
HUMAN_CONTENT_RATE = 0.015
TAIL_FRACTION = 0.30

WG, WL = 0.5, 0.5
ALPHA_SENS = 0.50
W_WIN = 15
GAMMA = 0.05
EPS = 1e-12

# ============================
# Figure autosave setup
# ============================
EXPERIMENT_NAME = f"Zipf_n{N_MODELS}_qS_ramp_KLJS"
FIG_DIR = os.path.join("figures", EXPERIMENT_NAME)
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {path}")

# ============================
# Zipf helpers
# ============================
zipf_base = 1.0 / np.power(ranks, ZIPF_S)
zipf_base = zipf_base / zipf_base.sum()

def make_zipf_gaussian(center_rank, width, base=None):
    base = zipf_base if base is None else base
    g = np.exp(-0.5 * ((ranks - center_rank) / width) ** 2)
    dist = base * g
    dist = np.clip(dist, 1e-12, None)
    return dist / dist.sum()

def make_zipf_variant(center_rank_shift=0.0, slope_delta=0.0, noise_scale=0.02):
    local_s = ZIPF_S + slope_delta
    shifted = np.clip(ranks + center_rank_shift, 1, None)
    base = 1.0 / np.power(shifted, local_s)
    noise = np.random.normal(0.0, noise_scale, size=base.shape)
    noisy = base * (1.0 + noise)
    noisy = np.clip(noisy, 1e-12, None)
    return noisy / noisy.sum()

# ============================
# Information theory helpers
# ============================
def shannon_entropy(p):
    p = np.clip(p, EPS, 1.0)
    return -np.sum(p * np.log(p))

def sliding_window_max_mean_negDlogD(D, W):
    s = -np.clip(D, EPS, 1.0) * np.log(np.clip(D, EPS, 1.0))
    if W <= 1:
        return float(np.max(s))
    c = np.concatenate([[0.0], np.cumsum(s)])
    window_sums = c[W:] - c[:-W]
    means = window_sums / float(W)
    return float(np.max(means))

def KL(p, q):
    p = np.clip(p, EPS, 1.0); q = np.clip(q, EPS, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)))

def JS(p, q):
    m = 0.5 * (p + q)
    return 0.5 * KL(p, m) + 0.5 * KL(q, m)

def q_global(D, H_H):
    H_D = shannon_entropy(D)
    return float(np.clip(1.0 - (H_D / (H_H + EPS)), 0.0, 1.0))

def q_local(D, Amax_star):
    A_t = sliding_window_max_mean_negDlogD(D, W_WIN)
    return float(np.clip((A_t / (Amax_star + EPS)) - 1.0, 0.0, 1.0))

def q_t(D, H_H, Amax_star):
    return float(np.clip(WG * q_global(D, H_H) + WL * q_local(D, Amax_star), 0.0, 1.0))

def S_gate(step, steps_total):
    if not START_SELF_LEARNING_LATE:
        return SELF_LEARNING
    t_start = int(SELF_LEARNING_START_FRACTION * steps_total)
    if step < t_start:
        return 0.0
    ramp_pos = step - t_start
    if ramp_pos >= SELF_LEARNING_RAMP_STEPS:
        return SELF_LEARNING
    return SELF_LEARNING * (ramp_pos / max(1, SELF_LEARNING_RAMP_STEPS))

def contam_from_f(f):
    w = f**2
    return w / (w.sum() + EPS)

# ============================
# Initialize datasets & models
# ============================
D_public = make_zipf_gaussian(center_rank=90, width=35)
D_ref = D_public.copy()
human_dist = make_zipf_gaussian(center_rank=220, width=80)
H = human_dist.copy()

H_H = shannon_entropy(H)
Amax_star = sliding_window_max_mean_negDlogD(H, W_WIN)
entropy_baseline = shannon_entropy(D_ref)

models = []
center_shifts = np.linspace(-25, 25, N_MODELS)
slope_deltas = np.linspace(-0.05, 0.05, N_MODELS)
for i in range(N_MODELS):
    shift = center_shifts[i] + np.random.uniform(-2, 2)
    slope = slope_deltas[i] + np.random.uniform(-0.01, 0.01)
    f_init = make_zipf_variant(center_rank_shift=10 + shift, slope_delta=-slope, noise_scale=0.02 + 0.002*i)
    D_init = make_zipf_variant(center_rank_shift=shift, slope_delta=slope, noise_scale=0.015 + 0.002*i)
    models.append({"f": f_init, "D": D_init})

y_target_og = models[0]["D"].copy()
y_second_target_og = D_public.copy()

tail_start = int((1.0 - TAIL_FRACTION) * VOCAB_SIZE)
baseline_tail_mass = np.sum(D_ref[tail_start:])

# ============================
# Diagnostics storage
# ============================
mse, kl_series, js_series = [], [], []
q_stagnation, q_tail_ratio, q_entropy_ratio = [], [], []
q_series, Hratio_series, S_series = [], [], []
H_xt = []
snapshots_all_models = [[] for _ in range(N_MODELS)]

# ============================
# Plot initial distributions
# ============================
plt.figure(figsize=(9,5))
base_colors = plt.cm.tab10(np.linspace(0,1,N_MODELS))
for i, m in enumerate(models):
    plt.plot(ranks, m["D"], lw=1.8, color=base_colors[i % 10], label=f"target {i+1} (2023)")
    plt.plot(ranks, m["f"], lw=1.2, linestyle="--", color=base_colors[i % 10], alpha=0.7, label=f"start {i+1}")
plt.plot(ranks, D_public, color="black", lw=2.5, label="public dataset D (2025)")
plt.xlabel("Token rank (1 = most frequent)")
plt.ylabel("Probability")
plt.title("Initial Zipfian Target and Start Distributions")
plt.legend(ncol=2, frameon=True)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
savefig("initial_distributions")
plt.show()

# ============================
# Iteration loop
# ============================
for step in range(STEPS):
    Sval = S_gate(step, STEPS)
    qval = q_t(D_public, H_H, Amax_star)
    coef = (MODEL_IMPROVEMENT_RATE + Sval) * (1.0 + ALPHA_SENS * qval)

    contaminations = []
    for m in models:
        m["f"] = m["f"] + coef * (D_public - m["f"])
        m["f"] = np.clip(m["f"], 1e-12, None); m["f"] /= m["f"].sum()
        contam_scale = Sval / DATASET_SIZE_RATIO
        C_f = contam_from_f(m["f"])
        contaminations.append(C_f)
        m["D"] = (1 - DATA_IMPROVEMENT_RATE - contam_scale) * m["D"] \
                 + DATA_IMPROVEMENT_RATE * D_public + contam_scale * C_f
        m["D"] = np.clip(m["D"], 1e-12, None); m["D"] /= m["D"].sum()

    C_avg = np.mean(contaminations, axis=0)
    D_public = D_public + HUMAN_CONTENT_RATE * H \
               + DATA_IMPROVEMENT_RATE * (H - D_public) \
               + GAMMA * Sval * C_avg
    D_public = np.clip(D_public, 1e-12, None); D_public /= D_public.sum()

    # ---- Ensemble-level diagnostics ----
    H_xt.append(-D_public * np.log(D_public + EPS))

    cosine_stag = np.mean([
        np.dot(m["f"], m["D"]) / (np.linalg.norm(m["f"]) * np.linalg.norm(m["D"]))
        for m in models
    ])

    entropy_curr = np.mean([-np.sum(m["f"] * np.log(m["f"] + EPS)) for m in models])
    entropy_ratio = entropy_curr / (entropy_baseline + EPS)

    tail_mass_curr = np.mean([np.sum(m["f"][tail_start:]) for m in models])
    tail_ratio = np.log10((tail_mass_curr + EPS) / (baseline_tail_mass + EPS))

    q_stagnation.append(cosine_stag)
    q_entropy_ratio.append(entropy_ratio)
    q_tail_ratio.append(tail_ratio)

    # Ensemble-average MSE and divergences
    f_mean = np.mean([m["f"] for m in models], axis=0)
    mse.append(np.mean((f_mean - D_ref) ** 2))
    kl_series.append(KL(f_mean, D_public))
    js_series.append(JS(f_mean, D_public))

    H_D = shannon_entropy(D_public)
    Hratio_series.append(H_D / (H_H + EPS))
    q_series.append(qval)
    S_series.append(Sval)

    if (step % PLOT_INTERVAL) == 0 or step == STEPS - 1:
        for j, m in enumerate(models):
            snapshots_all_models[j].append(m["f"].copy())

H_xt = np.array(H_xt)

# ============================
# Plots (autosaved)
# ============================

# Convergence plot (temporal overlay)
plt.figure(figsize=(9,6))
colors = plt.cm.viridis(np.linspace(0,1,N_MODELS))
for j, snapshots in enumerate(snapshots_all_models):
    for idx, dist in enumerate(snapshots):
        alpha = 0.05 + 0.75 * (idx / len(snapshots))
        plt.plot(ranks, dist, color=colors[j], alpha=alpha)
plt.plot(ranks, y_second_target_og, "k--", label="public 2025 dataset (ref)")
plt.xlabel("Token rank (1 = most frequent)")
plt.ylabel("Probability")
plt.title("Convergence of 10 Models under Zipfian Dynamics (Temporal Overlay)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True); plt.tight_layout()
savefig("convergence_all_models"); plt.show()

# MSE over time
plt.figure(figsize=(9,5))
plt.plot(mse, color="red")
plt.xlabel("Iteration"); plt.ylabel("MSE (ensemble vs ref)")
plt.title("MSE over Time")
plt.grid(True); plt.tight_layout()
savefig("mse_over_time"); plt.show()

# KL & JS
plt.figure(figsize=(9,5))
plt.plot(kl_series, label="KL(f_mean || D)", lw=1.5)
plt.plot(js_series, label="JS(f_mean, D)", lw=1.5)
plt.xlabel("Iteration"); plt.ylabel("Divergence")
plt.title("Model–Dataset Divergence (KL & JS)")
plt.grid(True); plt.legend(); plt.tight_layout()
savefig("kl_js_over_time"); plt.show()

# q(t), entropy ratio, and S(t)
fig, axs = plt.subplots(3,1, figsize=(10,9), sharex=True)
axs[0].plot(q_series); axs[0].set_ylabel('q(t)')
axs[1].plot(Hratio_series); axs[1].set_ylabel(r'$H_D/H_H$')
axs[2].plot(S_series); axs[2].set_ylabel('S(t)'); axs[2].set_xlabel('Iteration')
plt.suptitle('q(t), Entropy Ratio, and Self-Learning Gate')
plt.tight_layout(); savefig("q_entropy_S"); plt.show()

# Legacy diagnostics
plt.figure(figsize=(10,5))
plt.plot(q_stagnation, label="q_stagnation (avg model–dataset similarity)")
plt.plot(q_tail_ratio, label="q_tail_ratio (avg tail mass ratio)")
plt.plot(q_entropy_ratio, label="q_entropy_ratio (avg entropy ratio vs ref)")
plt.xlabel("Iteration"); plt.ylabel("Metric")
plt.title("Stagnation & Diversity Diagnostics (Ensemble)")
plt.legend(); plt.grid(True); plt.tight_layout()
savefig("stagnation_diagnostics"); plt.show()

# Entropy heatmaps
plt.figure(figsize=(10,6))
plt.imshow(H_xt.T, aspect='auto', cmap='plasma', origin='lower')
plt.xlabel('Iteration'); plt.ylabel('Token rank')
plt.title('Local Shannon Entropy Heatmap H(x,t)')
plt.colorbar(label='Entropy contribution')
plt.tight_layout(); savefig("entropy_heatmap"); plt.show()

dH_xt = H_xt[1:] - H_xt[:-1]
plt.figure(figsize=(10,6))
plt.imshow(dH_xt.T, aspect='auto', cmap='seismic', origin='lower', vmin=-0.001, vmax=0.001)
plt.xlabel('Iteration'); plt.ylabel('Token rank')
plt.title('ΔEntropy Map (Local Entropy Change)')
plt.colorbar(label='ΔH(x,t)')
plt.tight_layout(); savefig("delta_entropy_map"); plt.show()

# Phase diagram
plt.figure(figsize=(8,5))
plt.plot(q_series, Hratio_series, 'o-', alpha=0.6)
plt.xlabel('q(t)'); plt.ylabel(r'$H_D/H_H$')
plt.title('Phase Diagram of Diversity Decay')
plt.grid(True); plt.tight_layout()
savefig("phase_diagram"); plt.show()

# Entropy trendlines
plt.figure(figsize=(10,5))
for i in np.linspace(0, VOCAB_SIZE-1, 20, dtype=int):
    plt.plot(H_xt[:,i], alpha=0.6)
plt.title('Entropy Evolution of Sampled Token Ranks')
plt.xlabel('Iteration'); plt.ylabel('Local Entropy')
plt.grid(True); plt.tight_layout()
savefig("entropy_trendlines"); plt.show()

# ============================
# Persist metrics
# ============================
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

quality_df = pd.DataFrame({
    "q(t)": q_series,
    "H_ratio": Hratio_series,
    "S(t)": S_series,
    "MSE(ensemble_vs_ref)": mse,
    "KL(f_mean||D)": kl_series,
    "JS(f_mean,D)": js_series,
    "q_stagnation": q_stagnation,
    "q_tail_ratio": q_tail_ratio,
    "q_entropy_ratio": q_entropy_ratio
})
quality_df.to_csv(os.path.join(FIG_DIR, "quality_metrics.csv"), index=False)
