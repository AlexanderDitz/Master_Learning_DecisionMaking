import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load hidden states per trial
hidden_states_path = 'results/model_params/lstm_synthetic_data_hidden_states_per_trial.csv'
hidden_df = pd.read_csv(hidden_states_path)
hidden_cols = [col for col in hidden_df.columns if col.startswith('h_')]
assert len(hidden_cols) > 0, "No hidden state columns found!"

# Load Q-values and trial info
qvalues_path = 'data/synthetic_data/dezfouli2019_generated_behavior_lstm.csv'
q_df = pd.read_csv(qvalues_path)

# Find Q-value columns for action 0 and 1 (case-insensitive, allow underscores)
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
q0_col = find_q_col(q_df.columns, 0)
q1_col = find_q_col(q_df.columns, 1)
assert q0_col is not None and q1_col is not None, f"Could not find Q-value columns for actions 0 and 1. Found: {q0_col}, {q1_col}"

# Merge on trial index (assume both have same order or a 'trial' column)
if 'trial' in hidden_df.columns and 'trial' in q_df.columns:
    merged = pd.merge(hidden_df, q_df, on='trial', suffixes=('_h', '_q'))
    choice_col = 'choice_q' if 'choice_q' in merged.columns else 'choice'
    reward_col = 'reward_q' if 'reward_q' in merged.columns else 'reward'
else:
    # fallback: merge by index
    merged = pd.concat([hidden_df.add_suffix('_h'), q_df.add_suffix('_q')], axis=1)
    choice_col = 'choice_q'
    reward_col = 'reward_q'

# After merging, update Q-value column names to use suffixed names
q0_col = q0_col + '_q' if not q0_col.endswith('_q') else q0_col
q1_col = q1_col + '_q' if not q1_col.endswith('_q') else q1_col

# After merging, update hidden_cols to use suffixed names
hidden_cols = [col for col in merged.columns if col.startswith('h_') and col.endswith('_h')]
unit_labels = [col.replace('_h', '') for col in hidden_cols]

# Normalize hidden states to [-1, 1]
for col in hidden_cols:
    min_val = merged[col].min()
    max_val = merged[col].max()
    merged[col] = 2 * (merged[col] - min_val) / (max_val - min_val) - 1

combinations = [(0,0), (0,1), (1,0), (1,1)]
unit_labels = [f'h_{i}' for i in range(len(hidden_cols))]

for q, q_label in zip([q0_col, q1_col], ['q0', 'q1']):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(f'Correlation of LSTM unit state change with Δ{q_label} by action/reward', fontsize=16)
    for idx, (action, reward) in enumerate(combinations):
        ax = axes[idx//2, idx%2]
        mask = (merged[choice_col] == action) & (merged[reward_col] == reward)
        df = merged[mask].reset_index(drop=True)
        if len(df) < 2:
            ax.set_title(f'Choice {action}, Reward {reward} (Insufficient data)')
            ax.axis('off')
            continue
        # Compute state changes and Q-value changes
        hidden_t = df[hidden_cols].iloc[:-1].to_numpy()  # (n-1, n_units)
        hidden_t1 = df[hidden_cols].iloc[1:].to_numpy()  # (n-1, n_units)
        state_change = hidden_t1 - hidden_t              # (n-1, n_units)
        q_t = df[q].iloc[:-1].to_numpy()                # (n-1,)
        q_t1 = df[q].iloc[1:].to_numpy()                # (n-1,)
        dq = q_t1 - q_t                                 # (n-1,)
        # Correlate each unit's state change with dq
        corrs = [np.corrcoef(state_change[:,i], dq)[0,1] if np.std(state_change[:,i]) > 0 and np.std(dq) > 0 else 0 for i in range(state_change.shape[1])]
        sns.barplot(x=unit_labels, y=corrs, ax=ax, palette='viridis')
        ax.set_title(f'Choice {action}, Reward {reward}')
        ax.set_ylabel('Correlation')
        ax.set_xlabel('LSTM unit')
        ax.set_xticklabels(unit_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'results/vector_field_analysis/lstm_unit_corr_{q_label}.png', dpi=300, bbox_inches='tight')
    plt.show()
print(f'Analysis complete. Plots saved as lstm_unit_corr_{q0_col}.png and lstm_unit_corr_{q1_col}.png')
