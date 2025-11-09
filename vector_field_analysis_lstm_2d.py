import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter
import random
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
import re

# Set random state for reproducibility
np.random.seed(42)
random.seed(42)

# Set file paths directly
hidden_states_per_trial_path = 'lstm_synthetic_data_hidden_states_per_trial.csv'

# Load all hidden state columns automatically
hidden_df = pd.read_csv(hidden_states_per_trial_path)
hidden_cols = [col for col in hidden_df.columns if col.startswith('h_')]
assert len(hidden_cols) > 0, "No hidden state columns found!"

# Normalize hidden states to [-1, 1]
for col in hidden_cols:
    min_val = hidden_df[col].min()
    max_val = hidden_df[col].max()
    hidden_df[col] = 2 * (hidden_df[col] - min_val) / (max_val - min_val) - 1

# Load Q-values for correlation analysis
def find_q_col(cols, action):
    patterns = [
        rf'^q[_]?{action}$', rf'^Q[_]?{action}$', rf'^q{action}$', rf'^Q{action}$',
        rf'^q[_]?{action}[^\d]*', rf'^Q[_]?{action}[^\d]*'
    ]
    for pat in patterns:
        for col in cols:
            if re.match(pat, col):
                return col
    return None
qvalues_path = 'data/synthetic_data/dezfouli2019_generated_behavior_lstm.csv'
q_df = pd.read_csv(qvalues_path)
q0_col = find_q_col(q_df.columns, 0)
q1_col = find_q_col(q_df.columns, 1)
assert q0_col is not None and q1_col is not None, f"Could not find Q-value columns for actions 0 and 1. Found: {q0_col}, {q1_col}"

# Merge on trial index (assume both have same order or a 'trial' column)
if 'trial' in hidden_df.columns and 'trial' in q_df.columns:
    merged = pd.merge(hidden_df.add_suffix('_h'), q_df.add_suffix('_q'), left_on='trial_h', right_on='trial_q')
    choice_col = 'choice_q'
    reward_col = 'reward_q'
    hidden_cols_merged = [col for col in merged.columns if col.startswith('h_') and col.endswith('_h')]
else:
    merged = pd.concat([hidden_df.add_suffix('_h'), q_df.add_suffix('_q')], axis=1)
    choice_col = 'choice_q'
    reward_col = 'reward_q'
    hidden_cols_merged = [col for col in merged.columns if col.startswith('h_') and col.endswith('_h')]

# Define combinations, titles, and speed_cmap as in the original script
combinations = [(0,0), (0,1), (1,0), (1,1)]
titles = [
    r'Choice $_{0}$, Reward = 0',
    r'Choice $_{0}$, Reward = 1',
    r'Choice $_{1}$, Reward = 0',
    r'Choice $_{1}$, Reward = 1'
]
from matplotlib.colors import LinearSegmentedColormap
speed_cmap = LinearSegmentedColormap.from_list('speed_cmap', ['yellow', 'green', 'purple'])

# Identify key units for each action/reward combination (by |correlation| with ΔQ0 or ΔQ1)
key_units = {}
for idx, (action, reward) in enumerate(combinations):
    mask = (merged[choice_col] == action) & (merged[reward_col] == reward)
    df = merged[mask].reset_index(drop=True)
    if len(df) < 2:
        key_units[(action, reward)] = (hidden_cols_merged[0], hidden_cols_merged[1])
        continue
    hidden_t = df[hidden_cols_merged].iloc[:-1].to_numpy()
    hidden_t1 = df[hidden_cols_merged].iloc[1:].to_numpy()
    state_change = hidden_t1 - hidden_t
    q_col = q0_col + '_q' if action == 0 else q1_col + '_q'
    q_t = df[q_col].iloc[:-1].to_numpy()
    q_t1 = df[q_col].iloc[1:].to_numpy()
    dq = q_t1 - q_t
    corrs = [abs(np.corrcoef(state_change[:,i], dq)[0,1]) if np.std(state_change[:,i]) > 0 and np.std(dq) > 0 else 0 for i in range(state_change.shape[1])]
    top2 = np.argsort(corrs)[-2:][::-1]
    key_units[(action, reward)] = (hidden_cols_merged[top2[0]], hidden_cols_merged[top2[1]])

# --- Prepare subplots for all action/reward combinations, key units only ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')  # 2x2 grid
axes = axes.flatten()

# Collect all magnitudes for global color scaling
all_magnitudes = []
for idx, (action, reward) in enumerate(combinations):
    mask = (merged[choice_col] == action) & (merged[reward_col] == reward)
    df = merged[mask].reset_index(drop=True)
    if len(df) < 2:
        all_magnitudes.append(np.zeros((20, 20)))
        continue
    unit_x, unit_y = key_units[(action, reward)]
    x_t = df[unit_x].iloc[:-1].to_numpy()
    y_t = df[unit_y].iloc[:-1].to_numpy()
    x_t1 = df[unit_x].iloc[1:].to_numpy()
    y_t1 = df[unit_y].iloc[1:].to_numpy()
    dx = x_t1 - x_t
    dy = y_t1 - y_t
    n_grid = 20
    radius = 0.15
    grid = np.linspace(-1, 1, n_grid)
    G0, G1 = np.meshgrid(grid, grid)
    U = np.zeros_like(G0)
    V = np.zeros_like(G1)
    grid_points = np.stack([G0.ravel(), G1.ravel()], axis=1)
    data_points = np.stack([x_t, y_t], axis=1)
    for k, (g0, g1) in enumerate(grid_points):
        dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
        idxs = dist < radius
        if np.any(idxs):
            U.ravel()[k] = np.mean(dx[idxs])
            V.ravel()[k] = np.mean(dy[idxs])
        else:
            U.ravel()[k] = 0
            V.ravel()[k] = 0
    sigma = 1.0
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)
    magnitude = np.sqrt(U_smooth**2 + V_smooth**2)
    all_magnitudes.append(magnitude)
all_magnitudes = np.array(all_magnitudes)
global_vmin = all_magnitudes.min()
global_vmax = all_magnitudes.max()

# Plot with consistent color scaling and colorbar
for idx, (action, reward) in enumerate(combinations):
    ax = axes[idx]
    mask = (merged[choice_col] == action) & (merged[reward_col] == reward)
    df = merged[mask].reset_index(drop=True)
    n_trials = len(df)
    if n_trials < 2:
        ax.set_title(f"{titles[idx]} (Insufficient data)")
        ax.axis('off')
    else:
        unit_x, unit_y = key_units[(action, reward)]
        x_t = df[unit_x].iloc[:-1].to_numpy()
        y_t = df[unit_y].iloc[:-1].to_numpy()
        x_t1 = df[unit_x].iloc[1:].to_numpy()
        y_t1 = df[unit_y].iloc[1:].to_numpy()
        dx = x_t1 - x_t
        dy = y_t1 - y_t
        n_grid = 20
        radius = 0.15
        grid = np.linspace(-1, 1, n_grid)
        G0, G1 = np.meshgrid(grid, grid)
        U = np.zeros_like(G0)
        V = np.zeros_like(G1)
        grid_points = np.stack([G0.ravel(), G1.ravel()], axis=1)
        data_points = np.stack([x_t, y_t], axis=1)
        for k, (g0, g1) in enumerate(grid_points):
            dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
            idxs = dist < radius
            if np.any(idxs):
                U.ravel()[k] = np.mean(dx[idxs])
                V.ravel()[k] = np.mean(dy[idxs])
            else:
                U.ravel()[k] = 0
                V.ravel()[k] = 0
        sigma = 1.0
        U_smooth = gaussian_filter(U, sigma=sigma)
        V_smooth = gaussian_filter(V, sigma=sigma)
        magnitude = np.sqrt(U_smooth**2 + V_smooth**2)
        dxg = grid[1] - grid[0]
        grid_edges = np.linspace(-1 - dxg/2, 1 + dxg/2, n_grid + 1)
        bg = ax.pcolormesh(
            grid_edges, grid_edges, magnitude.T,
            cmap=speed_cmap, shading='auto', alpha=0.4, zorder=0,
            vmin=global_vmin, vmax=global_vmax
        )
        strm = ax.streamplot(
            grid, grid, U_smooth.T, V_smooth.T,
            color=magnitude.T,
            linewidth=1.2, cmap='viridis', density=1.2, arrowsize=1.5, zorder=2,
            norm=plt.Normalize(global_vmin, global_vmax)
        )
        ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.7, linewidth=2, zorder=4)
        ax.set_xlabel(f'{unit_x} (t)', fontsize=12)
        ax.set_ylabel(f'{unit_y} (t)', fontsize=12)
        ax.set_title(f"{titles[idx]} [n={n_trials}]", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_xticklabels(['-1', '1'])
        ax.set_yticklabels(['-1', '1'])
        ax.set_facecolor('white')
        ax.tick_params(axis='both', colors='black')
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
plt.tight_layout(pad=2.0, rect=[0, 0, 0.88, 1])  # leave space for colorbar on the right
# Add colorbar in a dedicated axes
cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(
    bg, cax=cbar_ax,
    orientation='vertical',
    label='Dynamics speed'
)
plt.savefig('vector_field_lstm.png', dpi=300, bbox_inches="tight")
plt.show()