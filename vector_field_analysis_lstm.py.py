import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set file paths directly
hidden_states_per_trial_path = 'lstm_synthetic_data_hidden_states_per_trial.csv'

# Define columns to load
cols_needed_hidden = ['participant', 'session', 'h_0', 'h_1', 'choice', 'reward']

# Load data
print(f"Loading: {hidden_states_per_trial_path}")
hidden_df = pd.read_csv(hidden_states_per_trial_path, usecols=cols_needed_hidden)

# Extract columns as numpy arrays
h0 = hidden_df['h_0'].to_numpy()
h1 = hidden_df['h_1'].to_numpy()
choice = hidden_df['choice'].to_numpy()
reward = hidden_df['reward'].to_numpy()

# Downsample by taking every n-th trial
step = 5
h0 = h0[::step]
h1 = h1[::step]
choice = choice[::step]
reward = reward[::step]

# Compute state changes
h0_change = h0[1:] - h0[:-1]
h1_change = h1[1:] - h1[:-1]

# --- Grid-based vector field computation (GQL-style, optimized) ---
n_grid = 8
radius = 0.1

h0_points = h0[:-1]
h1_points = h1[:-1]
h0_changes = h0_change
h1_changes = h1_change

h0_min, h0_max = np.min(h0), np.max(h0)
h1_min, h1_max = np.min(h1), np.max(h1)
h0_grid = np.linspace(h0_min, h0_max, n_grid)
h1_grid = np.linspace(h1_min, h1_max, n_grid)
H0, H1 = np.meshgrid(h0_grid, h1_grid)
U = np.zeros_like(H0)
V = np.zeros_like(H1)

# Vectorized computation over grid points
grid_points = np.stack([H0.ravel(), H1.ravel()], axis=1)
data_points = np.stack([h0_points, h1_points], axis=1)

for k, (g0, g1) in enumerate(grid_points):
    dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
    idx = dist < radius
    if np.any(idx):
        U.ravel()[k] = np.mean(h0_changes[idx])
        V.ravel()[k] = np.mean(h1_changes[idx])
    else:
        U.ravel()[k] = 0
        V.ravel()[k] = 0

# --- Trial-type split vector field plots ---
def extract_value_changes(output, value_type=0):
    x = output[:, value_type]
    x_change = x[1:] - x[:-1]
    return x[:-1], x_change

def plt_2d_vector_field_lstm(h0, h1, choice, reward, plot_n_decimal=1):
    if choice is None or reward is None:
        print('Error: choice or reward is None. Cannot split by trial type.')
        return
    output = np.stack([h0, h1], axis=1)
    choice_bin = np.array(choice[:-1])
    reward_bin = np.array(reward[:-1])
    if np.any(choice_bin == 2):
        choice_bin = choice_bin - 1
    if np.any(reward_bin > 1):
        reward_bin = (reward_bin > 0).astype(int)
    trial_types = choice_bin * 2 + reward_bin  # 0,1,2,3 for (A1,R0),(A1,R1),(A2,R0),(A2,R1)
    x1, x1_change = extract_value_changes(output, value_type=0)
    x2, x2_change = extract_value_changes(output, value_type=1)
    axis_max = max(np.max(x1), np.max(x2))
    axis_min = min(np.min(x1), np.min(x2))
    titles = ['A1 R=0', 'A1 R=1', 'A2 R=0', 'A2 R=1']
    print("Trial type counts:", [np.sum(trial_types == i) for i in range(4)])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    max_arrows = 10  # Maximum number of arrows to plot per subplot

    for trial_type in range(4):
        idx = trial_types == trial_type
        ax = axes[trial_type]
        if np.sum(idx) == 0:
            ax.set_title(titles[trial_type] + ' (empty)')
            print(f'Warning: No data for {titles[trial_type]}')
            continue
        # Downsample arrows
        arrow_indices = np.where(idx)[0]
        if len(arrow_indices) > max_arrows:
            arrow_indices = np.random.choice(arrow_indices, max_arrows, replace=False)
        # Black arrows: state changes (downsampled)
        ax.quiver(
            x1[arrow_indices], x2[arrow_indices],
            x1_change[arrow_indices], x2_change[arrow_indices],
            color='black', angles='xy', scale_units='xy', scale=1,
            alpha=0.8, width=0.004, headwidth=10, headlength=10, zorder=2
        )
        # Attractor states
        speed = np.sqrt(x1_change[idx]**2 + x2_change[idx]**2)
        attractor_idx = speed < np.percentile(speed, 10)
        ax.scatter(x1[idx][attractor_idx], x2[idx][attractor_idx], marker='x', color='white', s=80, zorder=3)
        # Indifference states
        epsilon = 0.05
        indiff_idx = np.abs(x1[idx] - x2[idx]) < epsilon
        ax.plot(x1[idx][indiff_idx], x2[idx][indiff_idx], linestyle='dashed', color='gray', linewidth=2, zorder=1)
        # Readout vector
        readout_vec = np.array([x1_change[idx].mean(), x2_change[idx].mean()])
        center = [np.mean(x1[idx]), np.mean(x2[idx])]
        ax.arrow(center[0], center[1], readout_vec[0], readout_vec[1], color='orange', width=0.01, head_width=0.08, head_length=0.08, length_includes_head=True, zorder=4)
        # Scatter: trial trajectory
        ax.scatter(x1[idx], x2[idx], c=np.arange(np.sum(idx)), cmap='viridis', s=20, zorder=5)
        ax.set_title(titles[trial_type])
        ax.set_xlabel('h0')
        ax.set_ylabel('h1')
        ax.set_xlim([axis_min, axis_max])
        ax.set_ylim([axis_min, axis_max])
        ax.set_aspect('equal')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Usage example (after extracting h0, h1, choice, reward):
plt_2d_vector_field_lstm(h0, h1, choice, reward)