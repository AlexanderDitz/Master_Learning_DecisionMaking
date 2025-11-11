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

# --- First loop: collect all magnitudes for global color scaling ---
all_magnitudes = []
for idx, (action, reward) in enumerate(combinations):
    mask = (hidden_df['choice'] == action) & (hidden_df['reward'] == reward)
    df = hidden_df[mask].reset_index(drop=True)
    if len(df) < 2:
        continue
    h0 = df['h_0'].to_numpy()
    h1 = df['h_1'].to_numpy()
    h0_change = h0[1:] - h0[:-1]
    h1_change = h1[1:] - h1[:-1]
    h0_points = h0[:-1]
    h1_points = h1[:-1]
    n_grid = 20
    radius = 0.15
    grid = np.linspace(-1, 1, n_grid)
    H0, H1 = np.meshgrid(grid, grid)
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
    h0 = df['h_0'].to_numpy()
    h1 = df['h_1'].to_numpy()
    h0_change = h0[1:] - h0[:-1]
    h1_change = h1[1:] - h1[:-1]
    h0_points = h0[:-1]
    h1_points = h1[:-1]
    n_grid = 20
    radius = 0.15
    grid = np.linspace(-1, 1, n_grid)
    H0, H1 = np.meshgrid(grid, grid)
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
    grid_edges = np.linspace(-1 - dx/2, 1 + dx/2, n_grid + 1)
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
        linewidth=1.2, cmap='viridis', density=1.2, arrowsize=1.5, zorder=2,
        norm=plt.Normalize(global_vmin, global_vmax)
    )
    # Dashed nullcline (diagonal for illustration)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.7, linewidth=2, zorder=4)
    # Axis formatting
    ax.set_xlabel('h_0', fontsize=12)
    ax.set_ylabel('h_1', fontsize=12)
    ax.set_title(f"{titles[idx]} [n={len(df)}]", fontsize=14)
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
plt.savefig('results/vector_field_rnn.png', dpi=300, bbox_inches="tight")
plt.show()
