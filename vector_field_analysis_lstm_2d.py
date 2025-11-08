import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import random

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

# --- Prepare subplots for all action/reward combinations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')
combinations = [(0,0), (0,1), (1,0), (1,1)]
titles = [
    r'Choice $_{0}$, Reward = 0',
    r'Choice $_{0}$, Reward = 1',
    r'Choice $_{1}$, Reward = 0',
    r'Choice $_{1}$, Reward = 1'
]

for idx, (action, reward) in enumerate(combinations):
    ax = axes[idx//2, idx%2]
    # Filter for this action/reward combination
    mask = (hidden_df['choice'] == action) & (hidden_df['reward'] == reward)
    df = hidden_df[mask].reset_index(drop=True)
    if len(df) < 2:
        ax.set_title(f"{titles[idx]} (Insufficient data)")
        ax.axis('off')
        continue

    # Compute mean hidden state per trial
    mean_hidden = df[hidden_cols].mean(axis=1).to_numpy()

    # Compute state changes (mean change per trial)
    mean_hidden_change = mean_hidden[1:] - mean_hidden[:-1]
    mean_hidden_t = mean_hidden[:-1]
    mean_hidden_t1 = mean_hidden[1:]

    # Prepare for grid-based vector field
    n_grid = 20
    radius = 0.15
    min_val, max_val = np.min(mean_hidden), np.max(mean_hidden)
    grid = np.linspace(-1, 1, n_grid)
    G0, G1 = np.meshgrid(grid, grid)
    U = np.zeros_like(G0)
    V = np.zeros_like(G1)
    grid_points = np.stack([G0.ravel(), G1.ravel()], axis=1)
    data_points = np.stack([mean_hidden_t, mean_hidden_t1], axis=1)
    for k, (g0, g1) in enumerate(grid_points):
        dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
        idxs = dist < radius
        if np.any(idxs):
            U.ravel()[k] = np.mean(mean_hidden_change[idxs])
            V.ravel()[k] = np.mean(mean_hidden_change[idxs])  # For 1D, use same for U and V
        else:
            U.ravel()[k] = 0
            V.ravel()[k] = 0

    # Smoothing
    sigma = 1.0
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)

    # Attractor detection
    def find_attractors(U, V, G0, G1, threshold=0.02):
        speed = np.sqrt(U**2 + V**2)
        attractor_idx = np.where(speed < threshold)
        attractor_points = np.column_stack((G0[attractor_idx], G1[attractor_idx]))
        return attractor_points
    attractor_points = find_attractors(U_smooth, V_smooth, G0, G1)

    # Plot
    ax.quiver(G0, G1, U_smooth, V_smooth, angles='xy', scale_units='xy', scale=2.5, alpha=0.95, color='royalblue', width=0.004, headwidth=2, headlength=3, zorder=3)
    if attractor_points.shape[0] > 0:
        ax.scatter(attractor_points[:, 0], attractor_points[:, 1], marker='x', color='red', s=40, linewidths=1.5, zorder=5)
    ax.set_xlabel('Mean hidden state (t)', fontsize=12)
    ax.set_ylabel('Mean hidden state (t+1)', fontsize=12)
    ax.set_title(titles[idx], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_facecolor('white')
    ax.tick_params(axis='both', colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

plt.tight_layout(pad=2.0)
plt.savefig('vector_field_grid_mean_hidden_lstm.png', dpi=300, bbox_inches="tight")
plt.show()