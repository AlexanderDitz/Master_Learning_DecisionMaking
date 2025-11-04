import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Plot 2D vector flow of Q-values for GQL/benchmark agent.')
parser.add_argument('--csv', type=str, required=False, default=None,
                    help='Path to the generated_behavior_benchmark.csv file')
parser.add_argument('--participant', type=str, required=False, default=None,
                    help='Participant ID to plot (default: first)')
parser.add_argument('--session', type=int, required=False, default=None,
                    help='Session index to plot (default: first)')
args = parser.parse_args()

# Auto-detect file if not provided
if args.csv is None:
    synthetic_dir = os.path.join('data', 'synthetic_data')
    args.csv = next((os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir)
                     if 'benchmark' in f and f.endswith('.csv')), None)
    if args.csv is None:
        raise FileNotFoundError('Could not find a benchmark synthetic data CSV file.')

def plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax=None, arrow_max_num=200, arrow_alpha=0.8, plot_n_decimal=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if len(x1) > arrow_max_num:
        idx = np.random.choice(len(x1), arrow_max_num, replace=False)
        x1, x1_change, x2, x2_change = x1[idx], x1_change[idx], x2[idx], x2_change[idx]
    ax.quiver(x1, x2, x1_change, x2_change, color=color,
              angles='xy', scale_units='xy', scale=1, alpha=arrow_alpha, width=0.004, headwidth=10, headlength=10)
    axis_min, axis_max = axis_range
    if axis_min < 0 < axis_max:
        axis_abs_max = max(abs(axis_min), abs(axis_max))
        axis_min, axis_max = -axis_abs_max, axis_abs_max
        ticks = [axis_min, 0, axis_max]
        ticklabels = [np.round(axis_min, plot_n_decimal), 0, np.round(axis_max, plot_n_decimal)]
    else:
        ticks = [axis_min, axis_max]
        ticklabels = [np.round(axis_min, plot_n_decimal), np.round(axis_max, plot_n_decimal)]
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_aspect('equal')
    return ax

def plot_generalized_q_vector_field_subplot(alpha=0.1, qmin=-1, qmax=1, grid_points=20):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    params = [
        (1, 0, 'Reward=1, Action=0'),
        (0, 0, 'Reward=0, Action=0'),
        (1, 1, 'Reward=1, Action=1'),
        (0, 1, 'Reward=0, Action=1'),
    ]
    for ax, (reward, action, title) in zip(axes.flat, params):
        Q0_grid, Q1_grid = np.meshgrid(
            np.linspace(qmin, qmax, grid_points),
            np.linspace(qmin, qmax, grid_points)
        )
        Q0_flat = Q0_grid.flatten()
        Q1_flat = Q1_grid.flatten()
        Q0_new = Q0_flat.copy()
        Q1_new = Q1_flat.copy()
        if action == 0:
            Q0_new += alpha * (reward - Q0_flat)
        else:
            Q1_new += alpha * (reward - Q1_flat)
        dQ0 = Q0_new - Q0_flat
        dQ1 = Q1_new - Q1_flat
        ax.quiver(Q0_flat, Q1_flat, dQ0, dQ1, angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Q0')
        ax.set_ylabel('Q1')
        ax.set_xlim([qmin, qmax])
        ax.set_ylim([qmin, qmax])
        ax.grid(True)
    plt.suptitle(f'Generalized Q-learning Vector Fields (alpha={alpha})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Load data and select participant/session
print(f"Loading: {args.csv}")
df = pd.read_csv(args.csv).dropna(subset=['Q0', 'Q1'])

# --- Individual participant/session plot (existing code) ---
pid = args.participant or df['id'].iloc[0]
sid = args.session or df[df['id'] == pid]['session'].iloc[0]
sub_df = df[(df['id'] == pid) & (df['session'] == sid)]
Q0, Q1 = sub_df['Q0'].values, sub_df['Q1'].values

# Assume your DataFrame has a 'reward' column (1 for rewarded, 0 for unrewarded)
rewarded = sub_df['reward'].values.astype(int)
Q0, Q1 = sub_df['Q0'].values, sub_df['Q1'].values

# Compute changes
Q0_change = Q0[1:] - Q0[:-1]
Q1_change = Q1[1:] - Q1[:-1]
rewarded = rewarded[:-1]  # Align with change arrays

# Split indices
idx_rewarded = rewarded == 1
idx_unrewarded = rewarded == 0

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
for ax, idx, label, color in zip(
    axes, [idx_rewarded, idx_unrewarded], ['Rewarded', 'Unrewarded'], ['green', 'red']):
    plt_2d_vector_flow(Q0[:-1][idx], Q0_change[idx], Q1[:-1][idx], Q1_change[idx], color=color,
                       axis_range=(min(Q0.min(), Q1.min()), max(Q0.max(), Q1.max())), ax=ax)
    ax.scatter(Q0[:-1][idx], Q1[:-1][idx], c=np.arange(np.sum(idx)), cmap='viridis', s=20, label=f'{label} Q trajectory')
    ax.set_title(f'Q-value Vector Flow ({label} Trials)')
    ax.set_xlabel('Q0')
    ax.set_ylabel('Q1')
    ax.legend()
plt.tight_layout()
plt.show()

# Compute vector field for Q0/Q1
x1 = Q0[:-1]
x2 = Q1[:-1]
x1_change = Q0[1:] - Q0[:-1]
x2_change = Q1[1:] - Q1[:-1]
axis_min = min(np.min(Q0), np.min(Q1))
axis_max = max(np.max(Q0), np.max(Q1))
axis_range = (axis_min, axis_max)
fig, ax = plt.subplots(figsize=(8, 8))
plt_2d_vector_flow(x1, x1_change, x2, x2_change, color='blue', axis_range=axis_range, ax=ax)
ax.scatter(Q0, Q1, c=range(len(Q0)), cmap='viridis', s=20, label='Q trajectory')
ax.set_xlabel('Q0')
ax.set_ylabel('Q1')
ax.set_title(f'Q-value Vector Flow (Participant {pid}, Session {sid})')
plt.colorbar(ax.collections[1], label='Trial', ax=ax)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# --- Mean Q0/Q1 vector field across all participants and sessions ---
print("\nPlotting mean Q-value vector field across all participants and sessions...")
# Find the minimum number of trials across all (participant, session) pairs
min_trials = df.groupby(['id', 'session']).size().min()

# Collect Q0 and Q1 arrays for all (participant, session) pairs, truncated to min_trials
all_Q0 = []
all_Q1 = []
for (pid, sid), group in df.groupby(['id', 'session']):
    q0 = group['Q0'].values[:min_trials]
    q1 = group['Q1'].values[:min_trials]
    if len(q0) == min_trials and len(q1) == min_trials:
        all_Q0.append(q0)
        all_Q1.append(q1)
all_Q0 = np.stack(all_Q0)
all_Q1 = np.stack(all_Q1)

mean_Q0 = all_Q0.mean(axis=0)
mean_Q1 = all_Q1.mean(axis=0)

# Compute mean vector field
x1 = mean_Q0[:-1]
x2 = mean_Q1[:-1]
x1_change = mean_Q0[1:] - mean_Q0[:-1]
x2_change = mean_Q1[1:] - mean_Q1[:-1]
axis_min = min(np.min(mean_Q0), np.min(mean_Q1))
axis_max = max(np.max(mean_Q0), np.max(mean_Q1))
axis_range = (axis_min, axis_max)
fig2, ax2 = plt.subplots(figsize=(8, 8))
plt_2d_vector_flow(x1, x1_change, x2, x2_change, color='red', axis_range=axis_range, ax=ax2)
ax2.scatter(mean_Q0, mean_Q1, c=range(len(mean_Q0)), cmap='plasma', s=20, label='Mean Q trajectory')
ax2.set_xlabel('Q0 (mean)')
ax2.set_ylabel('Q1 (mean)')
ax2.set_title('Mean Q-value Vector Flow (All Participants/Sessions)')
plt.colorbar(ax2.collections[1], label='Trial', ax=ax2)
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()

# Plot the generalized vector field with all possible action/reward combinations
plot_generalized_q_vector_field_subplot(alpha=0.1)