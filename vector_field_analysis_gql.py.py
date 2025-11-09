import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import random
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN

# Set random state for reproducibility
np.random.seed(42)
random.seed(42)

# Set file paths directly
q_values_per_trial_path = 'data/synthetic_data/dezfouli2019_generated_behavior_benchmark.csv'

# Load all Q-value columns automatically
q_df = pd.read_csv(q_values_per_trial_path)
assert 'Q0' in q_df.columns and 'Q1' in q_df.columns, "Q0 and Q1 columns required!"

# Normalize Q0 and Q1 to [0, 1]
for col in ['Q0', 'Q1']:
    min_val = q_df[col].min()
    max_val = q_df[col].max()
    q_df[col] = (q_df[col] - min_val) / (max_val - min_val)
    
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

def find_attractors(U, V, G0, G1, threshold=0.01, cluster_eps=0.08):
    speed = np.sqrt(U**2 + V**2)
    attractor_idx = np.where(speed < threshold)
    attractor_points = np.column_stack((G0[attractor_idx], G1[attractor_idx]))
    if len(attractor_points) > 0:
        clustering = DBSCAN(eps=cluster_eps, min_samples=1).fit(attractor_points)
        attractor_points = np.array([attractor_points[clustering.labels_ == i].mean(axis=0)
                                     for i in np.unique(clustering.labels_)])
    return attractor_points

for idx, (action, reward) in enumerate(combinations):
    ax = axes[idx//2, idx%2]
    # Filter for this action/reward combination
    mask = (q_df['choice'] == action) & (q_df['reward'] == reward)
    df = q_df[mask].reset_index(drop=True)
    if len(df) < 2:
        ax.set_title(f"{titles[idx]} (Insufficient data)")
        ax.axis('off')
        continue

    # Use Q0 and Q1 for 2D vector field
    q0 = df['Q0'].to_numpy()
    q1 = df['Q1'].to_numpy()

    # Compute state changes (trial to trial)
    dq0 = q0[1:] - q0[:-1]
    dq1 = q1[1:] - q1[:-1]
    q0_points = q0[:-1]
    q1_points = q1[:-1]

    # Prepare for grid-based vector field
    n_grid = 20
    radius = 0.05
    grid = np.linspace(0, 1, n_grid)
    G0, G1 = np.meshgrid(grid, grid)
    U = np.zeros_like(G0)
    V = np.zeros_like(G1)
    grid_points = np.stack([G0.ravel(), G1.ravel()], axis=1)
    data_points = np.stack([q0_points, q1_points], axis=1)
    for k, (g0, g1) in enumerate(grid_points):
        dist = np.sqrt((data_points[:, 0] - g0)**2 + (data_points[:, 1] - g1)**2)
        idxs = dist < radius
        if np.any(idxs):
            U.ravel()[k] = np.mean(dq0[idxs])
            V.ravel()[k] = np.mean(dq1[idxs])
        else:
            U.ravel()[k] = 0
            V.ravel()[k] = 0

    # Smoothing
    sigma = 1.0
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)

    # Attractor detection (with clustering)
    attractor_points = find_attractors(U_smooth, V_smooth, G0, G1, threshold=0.01, cluster_eps=0.08)

    # Compute magnitude for coloring (dynamics speed)
    magnitude = np.sqrt(U_smooth**2 + V_smooth**2)

    # Compute the edges of the grid for pcolormesh
    dx = grid[1] - grid[0]
    grid_edges = np.linspace(0 - dx/2, 1 + dx/2, n_grid + 1)

    # Plot background as dynamics speed
    bg = ax.pcolormesh(
        grid_edges, grid_edges, magnitude.T,
        cmap=speed_cmap, shading='auto', alpha=0.7, zorder=0
    )

    # Add streamlines for flow visualization (flowlines only)
    strm = ax.streamplot(
        grid, grid, U_smooth.T, V_smooth.T,
        color=magnitude.T,
        linewidth=1.2, cmap='viridis', density=1.2, arrowsize=1.0, zorder=2
    )

    # Attractors as white crosses
    if attractor_points.shape[0] > 0:
        ax.scatter(attractor_points[:, 0], attractor_points[:, 1], marker='x', color='white', s=60, linewidths=2, zorder=5)

    # Dashed nullcline (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, zorder=4)

    # Orange readout vector (mean state change)
    mean_q0 = np.mean(q0_points)
    mean_q1 = np.mean(q1_points)
    mean_dq0 = np.mean(dq0)
    mean_dq1 = np.mean(dq1)
    ax.arrow(mean_q0, mean_q1, mean_dq0, mean_dq1, color='orange', width=0.008, head_width=0.04, head_length=0.04, length_includes_head=True, zorder=6)

    # Axis formatting
    ax.set_xlabel('Q0 (t)', fontsize=12)
    ax.set_ylabel('Q1 (t)', fontsize=12)
    ax.set_title(f"{titles[idx]}   [n={len(df)}]", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')
    ax.tick_params(axis='both', colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

plt.subplots_adjust(right=0.85)
cbar = plt.colorbar(
    bg, ax=axes.ravel().tolist(),
    orientation='vertical',
    fraction=0.025, pad=0.04, label='Dynamics speed'
)

plt.tight_layout(pad=2.0)
plt.savefig('vector_field_grid_benchmark.png', dpi=300, bbox_inches="tight")
plt.show()