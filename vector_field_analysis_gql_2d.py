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

all_magnitudes = []

# --- First loop: collect all magnitudes for global color scaling ---
for idx, (action, reward) in enumerate(combinations):
    mask = (q_df['choice'] == action) & (q_df['reward'] == reward)
    df = q_df[mask].reset_index(drop=True)
    if len(df) < 2:
        continue
    q0 = df['Q0'].to_numpy()
    q1 = df['Q1'].to_numpy()
    dq0 = q0[1:] - q0[:-1]
    dq1 = q1[1:] - q1[:-1]
    q0_points = q0[:-1]
    q1_points = q1[:-1]
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
    mask = (q_df['choice'] == action) & (q_df['reward'] == reward)
    df = q_df[mask].reset_index(drop=True)
    if len(df) < 2:
        ax.set_title(f"{titles[idx]} (Insufficient data)")
        ax.axis('off')
        continue
    q0 = df['Q0'].to_numpy()
    q1 = df['Q1'].to_numpy()
    dq0 = q0[1:] - q0[:-1]
    dq1 = q1[1:] - q1[:-1]
    q0_points = q0[:-1]
    q1_points = q1[:-1]
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
    sigma = 1.0
    U_smooth = gaussian_filter(U, sigma=sigma)
    V_smooth = gaussian_filter(V, sigma=sigma)
    magnitude = np.sqrt(U_smooth**2 + V_smooth**2)
   
    # Summary statistics for flow
    mean_speed = np.mean(magnitude)
    mean_U = np.mean(U_smooth)
    mean_V = np.mean(V_smooth)
    angle = np.arctan2(mean_V, mean_U) * 180 / np.pi  # in degrees

    print(f"Action {action}, Reward {reward}:")
    print(f"  Mean speed: {mean_speed:.3f}")
    print(f"  Mean direction: ({mean_U:.3f}, {mean_V:.3f}), angle: {angle:.1f}°")

    dx = grid[1] - grid[0]
    grid_edges = np.linspace(0 - dx/2, 1 + dx/2, n_grid + 1)
    # Speedmap background
    bg = ax.pcolormesh(
        grid_edges, grid_edges, magnitude.T,
        cmap=speed_cmap, shading='auto', alpha=0.4, zorder=0,
        vmin=global_vmin, vmax=global_vmax
    )
    # Streamlines (flow lines)
    strm = ax.streamplot(
        grid, grid, U_smooth.T, V_smooth.T,
        color=magnitude.T,
        linewidth=1.2, cmap='viridis', density=1.2, arrowsize=1.0, zorder=2,
        norm=plt.Normalize(global_vmin, global_vmax)
    )
    # Dashed nullcline (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, zorder=4)
    # Axis formatting
    ax.set_xlabel('Q0 (t)', fontsize=12)
    ax.set_ylabel('Q1 (t)', fontsize=12)
    ax.set_title(f"{titles[idx]}   [n={len(df)}]", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_facecolor('white')
    ax.tick_params(axis='both', colors='black')
    # Only show 0 and 1 as axis ticks (integers)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=12)
    ax.set_yticklabels(["0", "1"], fontsize=12)
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
plt.savefig('vector_field_benchmark.png', dpi=300, bbox_inches="tight")
plt.show()