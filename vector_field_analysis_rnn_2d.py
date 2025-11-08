import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# Set file paths directly
hidden_states_per_trial_path = 'rnn_synthetic_data_hidden_states_per_trial.csv'

# Define columns to load
cols_needed_hidden = ['h_0', 'h_1', 'choice', 'reward']

# Load data (pooling all participants/sessions)
print(f"Loading: {hidden_states_per_trial_path}")
hidden_df = pd.read_csv(hidden_states_per_trial_path, usecols=cols_needed_hidden)

for col in ['h_0', 'h_1']:
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

# Custom colormap: yellow (slow), green (medium), purple (fast)
speed_cmap = LinearSegmentedColormap.from_list('speed_cmap', ['yellow', 'green', 'purple'])

for idx, (action, reward) in enumerate(combinations):
    ax = axes[idx//2, idx%2]
    # Filter for this action/reward combination
    mask = (hidden_df['choice'] == action) & (hidden_df['reward'] == reward)
    df = hidden_df[mask].reset_index(drop=True)
    if len(df) < 2:
        ax.set_title(f"{titles[idx]} (Insufficient data)")
        ax.axis('off')
        continue
    h0 = df['h_0'].to_numpy()
    h1 = df['h_1'].to_numpy()
    # Compute state changes
    h0_change = h0[1:] - h0[:-1]
    h1_change = h1[1:] - h1[:-1]
    h0_points = h0[:-1]
    h1_points = h1[:-1]
    # --- Grid-based vector field computation with smoothing ---
    n_grid = 20
    radius = 0.15
    h0_min, h0_max = np.min(h0), np.max(h0)
    h1_min, h1_max = np.min(h1), np.max(h1)
    h0_grid = np.linspace(h0_min, h0_max, n_grid)
    h1_grid = np.linspace(h1_min, h1_max, n_grid)
    H0, H1 = np.meshgrid(h0_grid, h1_grid)
    U = np.zeros_like(H0)
    V = np.zeros_like(H1)
    grid_points = np.stack([H0.ravel(), H1.ravel()], axis=1)
    data_points = np.stack([h0_points, h1_points], axis=1)
    for k, (g0, g1) in enumerate(grid_points):
        dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
        idxs = dist < radius
        if np.any(idxs):
            U.ravel()[k] = np.mean(h0_change[idxs])
            V.ravel()[k] = np.mean(h1_change[idxs])
        else:
            U.ravel()[k] = 0
            V.ravel()[k] = 0
    # --- Apply Gaussian smoothing to the vector field ---
    sigma = 1.0  # Increase for more smoothing
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)
    # --- Attractor/Nullcline Example (replace with your own logic) ---
    def find_attractors(U, V, H0, H1, threshold=0.02):
        speed = np.sqrt(U**2 + V**2)
        attractor_idx = np.where(speed < threshold)
        attractor_points = np.column_stack((H0[attractor_idx], H1[attractor_idx]))
        return attractor_points
    attractor_points = find_attractors(U_smooth, V_smooth, H0, H1)
    # --- Compute speed for background ---
    speed = np.sqrt(U_smooth**2 + V_smooth**2)
    speed_norm = (speed - np.min(speed)) / (np.max(speed) - np.min(speed) + 1e-8)
    # --- Plot vector field (blue, finer arrows) ---
    ax.quiver(H0, H1, U_smooth, V_smooth, angles='xy', scale_units='xy', scale=2.5, alpha=0.95, color='royalblue', width=0.004, headwidth=2, headlength=3, zorder=3)
    # --- Attractors: small bright red crosses ---
    if attractor_points.shape[0] > 0:
        ax.scatter(attractor_points[:, 0], attractor_points[:, 1], marker='x', color='red', s=40, linewidths=1.5, zorder=5)
    # --- Dashed nullcline (diagonal for illustration) ---
    ax.plot([h0_min, h0_max], [h1_min, h1_max], 'k--', alpha=0.7, linewidth=2, zorder=4)
    # --- Orange readout vector (mean state change) ---
    mean_h0 = np.mean(h0_points)
    mean_h1 = np.mean(h1_points)
    mean_dh0 = np.mean(h0_change)
    mean_dh1 = np.mean(h1_change)
    ax.arrow(mean_h0, mean_h1, mean_dh0, mean_dh1, color='orange', width=0.008, head_width=0.06, head_length=0.06, length_includes_head=True, zorder=6)
    # --- Axis formatting ---
    ax.set_xlabel('h_0', fontsize=12)
    ax.set_ylabel('h_1', fontsize=12)
    ax.set_title(titles[idx], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_facecolor('white')
    ax.tick_params(axis='both', colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

plt.tight_layout(pad=2.0)
plt.savefig('vector_field_grid.png', dpi=300, bbox_inches="tight")
plt.show()
