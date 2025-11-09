import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter
import random
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN

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

# Custom colormap: yellow (slow), green (medium), purple (fast)
speed_cmap = LinearSegmentedColormap.from_list('speed_cmap', ['yellow', 'green', 'purple'])

def find_attractors(U, V, G0, G1, threshold=0.01, cluster_eps=0.12):
    speed = np.sqrt(U**2 + V**2)
    attractor_idx = np.where(speed < threshold)
    attractor_points = np.column_stack((G0[attractor_idx], G1[attractor_idx]))
    if len(attractor_points) > 0:
        clustering = DBSCAN(eps=cluster_eps, min_samples=1).fit(attractor_points)
        attractor_points = np.array([attractor_points[clustering.labels_ == i].mean(axis=0)
                                     for i in np.unique(clustering.labels_)])
    return attractor_points

# --- First loop: collect all magnitudes for global color scaling ---
all_magnitudes = []
for idx, (action, reward) in enumerate(combinations):
    mask = (hidden_df['choice'] == action) & (hidden_df['reward'] == reward)
    df = hidden_df[mask].reset_index(drop=True)
    if len(df) < 2:
        continue
    mean_hidden = df[hidden_cols].mean(axis=1).to_numpy()
    mean_hidden_t = mean_hidden[:-1]
    mean_hidden_t2 = mean_hidden[:-1]  # Use t for both axes for grid
    d_hidden = mean_hidden[1:] - mean_hidden[:-1]
    n_grid = 20
    radius = 0.15
    grid = np.linspace(-1, 1, n_grid)
    G0, G1 = np.meshgrid(grid, grid)
    U = np.zeros_like(G0)
    V = np.zeros_like(G1)
    grid_points = np.stack([G0.ravel(), G1.ravel()], axis=1)
    data_points = np.stack([mean_hidden_t, mean_hidden_t2], axis=1)
    for k, (g0, g1) in enumerate(grid_points):
        dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
        idxs = dist < radius
        if np.any(idxs):
            U.ravel()[k] = np.mean(d_hidden[idxs])
            V.ravel()[k] = np.mean(d_hidden[idxs])
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

# --- Second loop: plot with consistent color scaling and colorbar ---
for idx, (action, reward) in enumerate(combinations):
    ax = axes[idx//2, idx%2]
    mask = (hidden_df['choice'] == action) & (hidden_df['reward'] == reward)
    df = hidden_df[mask].reset_index(drop=True)
    if len(df) < 2:
        ax.set_title(f"{titles[idx]} (Insufficient data)")
        ax.axis('off')
        continue
    mean_hidden = df[hidden_cols].mean(axis=1).to_numpy()
    mean_hidden_t = mean_hidden[:-1]
    mean_hidden_t2 = mean_hidden[:-1]  # Use t for both axes for grid
    d_hidden = mean_hidden[1:] - mean_hidden[:-1]
    n_grid = 20
    radius = 0.15
    grid = np.linspace(-1, 1, n_grid)
    G0, G1 = np.meshgrid(grid, grid)
    U = np.zeros_like(G0)
    V = np.zeros_like(G1)
    grid_points = np.stack([G0.ravel(), G1.ravel()], axis=1)
    data_points = np.stack([mean_hidden_t, mean_hidden_t2], axis=1)
    for k, (g0, g1) in enumerate(grid_points):
        dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
        idxs = dist < radius
        if np.any(idxs):
            U.ravel()[k] = np.mean(d_hidden[idxs])
            V.ravel()[k] = np.mean(d_hidden[idxs])
        else:
            U.ravel()[k] = 0
            V.ravel()[k] = 0
    sigma = 1.0
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)
    attractor_points = find_attractors(U_smooth, V_smooth, G0, G1, threshold=0.01, cluster_eps=0.12)
    magnitude = np.sqrt(U_smooth**2 + V_smooth**2)
    grid_edges = np.linspace(-1, 1, n_grid + 1)
    # Plot background as dynamics speed
    bg = ax.pcolormesh(
        grid_edges, grid_edges, magnitude.T,
        cmap=speed_cmap, shading='auto', alpha=0.4, zorder=0,
        vmin=global_vmin, vmax=global_vmax
    )
    # Add streamlines for flow visualization (flowlines only)
    strm = ax.streamplot(
        grid, grid, U_smooth.T, V_smooth.T,
        color=magnitude.T,
        linewidth=1.2, cmap='viridis', density=1.2, arrowsize=1.0, zorder=2,
        norm=plt.Normalize(global_vmin, global_vmax)
    )
    # Dashed nullcline (diagonal)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.7, linewidth=2, zorder=4)
    # Axis formatting
    ax.set_xlabel('Mean hidden state (t)', fontsize=14)
    ax.set_ylabel('Mean hidden state (t)', fontsize=14)
    ax.set_title(f"{titles[idx]}   [n={len(df)}]", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_facecolor('white')
    ax.tick_params(axis='both', labelsize=12, colors='black')
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Only show -1 and 1 on axes, as integers
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_xticklabels(['-1', '1'], fontsize=12)
    ax.set_yticklabels(['-1', '1'], fontsize=12)

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