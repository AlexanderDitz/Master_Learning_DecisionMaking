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

# Load data and select participant/session
print(f"Loading: {args.csv}")
df = pd.read_csv(args.csv).dropna(subset=['Q0', 'Q1'])
pid = args.participant or df['id'].iloc[0]
sid = args.session or df[df['id'] == pid]['session'].iloc[0]
sub_df = df[(df['id'] == pid) & (df['session'] == sid)]
Q0, Q1 = sub_df['Q0'].values, sub_df['Q1'].values

print(f"Q0: min={Q0.min()}, max={Q0.max()}, unique={np.unique(Q0)}")
print(f"Q1: min={Q1.min()}, max={Q1.max()}, unique={np.unique(Q1)}")
print(f"Number of trials: {len(Q0)}")

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
