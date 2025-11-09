import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

STEPS = 200                                 # number of iterations to run the convergence process
MODEL_IMPROVEMENT_RATE = 0.02               # how fast the model approaches the current dataset
SELF_LEARNING = 0.02                        # how fast the model expands its own data set
DATA_IMPROVEMENT_RATE = 0.05                # how fast the dataset approaches the final target
PLOT_INTERVAL = 5                           # how often to plot the current state of the model
START_SELF_LEARNING_LATE = True             # whether to start self learning only after a certain fraction of the process
SELF_LEARNING_START_FRACTION = 0.25         # when to start self learning (if START_SELF_LEARNING_LATE is True)

# target_func = lambda x: 0.5*np.exp(-1*(x-1.2))
target_func = lambda x, mu=0.0, sigma=1.0: (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((3*x - mu) / sigma) ** 2)
# target_func = lambda x: np.sin(3*x) + np.cos(1*x) + 2
# target_func = lambda x: np.sin(3*x) + np.cos(1*x) + 2 + np.random.rand(*x.shape) * 5

# start_func = lambda x: np.ones_like(x)
start_func = lambda x, mu=2.5, sigma=1.0: 2 * (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((3*x - mu) / sigma) ** 2)
# start_func = lambda x: np.random.rand(*x.shape)
# start_func = lambda x, mu=0.0, sigma=1.0: (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((2*x - mu) / sigma) ** 2)
# start_func = lambda x: np.sin(5*x) + np.cos(2*x) + 2




# second_target = lambda x: 0.5*np.exp(-2*(x+1.2))
second_target = lambda x, mu=-2.5, sigma=1.0: 2 * (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((3*x - mu) / sigma) ** 2)
# second_target = lambda x: np.cos(3*(x-1)) + np.sin(2*x) + 2
# second_target = lambda x: np.cos(3*(x-1)) + np.sin(2*x) + 2 + np.random.rand(*x.shape) * 5
# second_target = lambda x, mu=0.0, sigma=1.0: (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((5*x - mu) / sigma) ** 2)


x = np.linspace(-2, 2, 400)

y_target = target_func(x)
y_target = y_target / np.sum(y_target)  # Normalize to sum to 1

y_start = start_func(x)
y_start = y_start / np.sum(y_start)  # Normalize to sum to 1

y_second_target = second_target(x)
y_second_target = y_second_target / np.sum(y_second_target)  # Normalize to sum to 1

y_target_og = y_target.copy()
y_start_og = y_start.copy()
y_second_target_og = y_second_target.copy()

mse = []

plt.figure(figsize=(8, 5))
plt.plot(x, y_target, color="orange", label="target func")
plt.plot(x, y_start, color="blue", label="start func")
plt.plot(x, y_second_target, color="black", label="second target")
plt.legend()
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Target and Start Functions')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
for step in range(STEPS):
    if START_SELF_LEARNING_LATE and step / STEPS < SELF_LEARNING_START_FRACTION:
        self_learning_rate = 0.0
    else:
        self_learning_rate = SELF_LEARNING
    y_start = (1 - MODEL_IMPROVEMENT_RATE - self_learning_rate) * y_start + (MODEL_IMPROVEMENT_RATE + self_learning_rate) * y_target
    y_target = y_target / np.sum(y_target)  # Normalize to sum to 1

    y_target = (1 - DATA_IMPROVEMENT_RATE - self_learning_rate) * y_target + DATA_IMPROVEMENT_RATE * y_second_target + self_learning_rate * (y_start * y_start)/np.sum(y_start * y_start)
    y_start = y_start / np.sum(y_start)  # Normalize to sum to 1

    y_second_target = (1 - self_learning_rate) * y_second_target + self_learning_rate * (y_start * y_start)/np.sum(y_start * y_start)
    y_second_target = y_second_target / np.sum(y_second_target)  # Normalize to sum to 1

    mse.append(np.mean((y_start - y_second_target_og) ** 2))

    if (step % PLOT_INTERVAL) == 0:
        plt.plot(x, y_start, color="blue", alpha = (step/STEPS))

# plt.plot(x, y_target, color="orange", label=f"current data")
# plt.plot(x, y_second_target, color="black", label=f"final data target")
plt.plot(x, y_target_og, color="orange", linestyle="--", label=f"original target")
plt.plot(x, y_second_target_og, color="black", linestyle="--", label=f"original second target")
plt.legend()
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Convergence of Start to Target')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(mse, color="red")
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('MSE between Start and Second Target')
plt.ylim(bottom=0)
plt.grid()
plt.show()

csv_path = "mse.csv"
col_name = str(SELF_LEARNING)  # column name based on the SELF_LEARNING value
df_new = pd.DataFrame({col_name: mse})

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    max_len = max(len(df), len(df_new))
    df = df.reindex(range(max_len))
    df[col_name] = pd.Series(mse)
else:
    df = df_new

df.to_csv(csv_path, index=False)

if os.path.exists(csv_path):
    df_plot = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    for col in df_plot.columns:
        series = df_plot[col]
        mask = series.notna()
        plt.plot(df_plot.index[mask], series[mask], label=col)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSEs from CSV')
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend(title='SELF_LEARNING')
    plt.show()
else:
    print(f"{csv_path} not found, nothing to plot.")