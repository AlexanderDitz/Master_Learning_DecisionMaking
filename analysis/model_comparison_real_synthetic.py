import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from itertools import combinations
from scipy import stats

# ----------------- Setup -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
pd.set_option('future.no_silent_downcasting', True)

# ----------------- Load Data -----------------

# Load synthetic RNN (l2 = 0.001) data
rnn_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_rnn_l2_0_001.csv'
rnn_df = pd.read_csv(rnn_file)
rnn_df['model_type'] = 'RNN'

# Load synthetic RNN (l2 = 0.0001) data
# rnn2_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_rnn2_l2_0_0001.csv'
# rnn2_df = pd.read_csv(rnn2_file)
# rnn2_df['model_type'] = 'RNN2'

# Load synthetic RNN (l2 = 0.00001) data
# rnn3_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_rnn3_l2_0_00001.csv'
# rnn3_df = pd.read_csv(rnn3_file)
# rnn3_df['model_type'] = 'RNN3'

# Load synthetic RNN (l2 = 0.0005) data
# rnn4_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_rnn4_l2_0_0005.csv'
# rnn4_df = pd.read_csv(rnn4_file)
# rnn4_df['model_type'] = 'RNN4'

# Load synthetic RNN (l2 = 0.00005) data
# rnn5_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_rnn5_l2_0_00005.csv'
# rnn5_df = pd.read_csv(rnn5_file)
# rnn5_df['model_type'] = 'RNN5'

# Load synthetic SPICE (l2 = 0.001) model data
spice2_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice2_l2_0_001.csv'
spice2_df = pd.read_csv(spice2_file)
spice2_df['model_type'] = 'SPICE2'

# Load synthetic SPICE (l2 = 0.0001) model data
# spice3_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice3_l2_0_0001.csv'
# spice3_df = pd.read_csv(spice3_file)
# spice3_df['model_type'] = 'SPICE3'

# Load synthetic SPICE (l2 = 0.00001) model data
# spice4_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice4_l2_0_00001.csv'
# spice4_df = pd.read_csv(spice4_file)
# spice4_df['model_type'] = 'SPICE4'

# Load synthetic SPICE (l2 = 0.0005) model data
# spice5_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice5_l2_0_0005.csv'
# spice5_df = pd.read_csv(spice5_file)
# spice5_df['model_type'] = 'SPICE5'

# Load synthetic SPICE (l2 = 0.00005) model data
# spice6_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice6_l2_0_00005.csv'
# spice6_df = pd.read_csv(spice6_file)
# spice6_df['model_type'] = 'SPICE6'

# Load benchmark model data
benchmark_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_benchmark.csv'
benchmark_df = pd.read_csv(benchmark_file)
benchmark_df['model_type'] = 'BENCHMARK'

# Load LSTM model data
lstm_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_lstm.csv'
lstm_df = pd.read_csv(lstm_file)
lstm_df['model_type'] = 'LSTM'

# Load Q-learning model data
q_file = '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_q_agent.csv'
q_df = pd.read_csv(q_file)
q_df['model_type'] = 'Q_AGENT'

# Load real participant data
real_file = '../data/features/real_features/real_participant_features.csv'
real_df = pd.read_csv(real_file)
real_df['model_type'] = 'HUMAN'

# Combine data
df = pd.concat([real_df, rnn_df, spice2_df, benchmark_df, lstm_df, q_df], ignore_index=True)

# ----------------- Configuration -----------------
model_order = ['HUMAN', 'RNN', 'SPICE2', 'BENCHMARK', 'LSTM', 'Q_AGENT']
model_colors = {'HUMAN': '#2E8B57', 'RNN': '#FF4500', 'SPICE2': '#FF6347', 'BENCHMARK': '#7B68EE', 'LSTM': '#4682B4', 'Q_AGENT': '#5F9EA0'}
features = ['choice_rate', 'reward_rate', 'win_stay', 'win_shift', 'lose_stay', 'lose_shift', 'choice_perseveration', 'switch_rate']
feature_labels = {
    'choice_rate': 'Choice Rate',
    'reward_rate': 'Reward Rate',
    'win_stay': 'Win-Stay Rate',
    'win_shift': 'Win-Shift Rate',
    'lose_stay': 'Lose-Stay Rate',
    'lose_shift': 'Lose-Shift Rate',
    'choice_perseveration': 'Choice Perseveration',
    'switch_rate': 'Switch Rate'
}

os.makedirs('../data/visualization_plots', exist_ok=True)

# ----------------- Boxplots with points -----------------
plot_counter = 12  # Start plot counter at 10
sns.set_style("whitegrid")

for feature in features:
    plt.figure(figsize=(10, 6))
    present_models = [m for m in model_order if m in df['model_type'].unique()]
    present_colors = [model_colors[m] for m in present_models]
    
    # Boxplot
    sns.boxplot(data=df, x='model_type', y=feature, order=present_models, palette=present_colors,
                showfliers=False, width=0.6)
    
    # Overlay points
    for i, model in enumerate(present_models):
        group_data = df[df['model_type'] == model][feature]
        x_positions = np.random.normal(i, 0.08, size=len(group_data))
        plt.scatter(x_positions, group_data, color=model_colors[model], edgecolor='white', s=30, alpha=0.8)
    
    plt.xlabel('Model Type')
    plt.ylabel(feature_labels[feature])
    plt.title(f'{feature_labels[feature]} by Model')
    plt.xticks(range(len(present_models)), [f'{m}\n(n={len(df[df["model_type"]==m])})' for m in present_models])
    plt.tight_layout()
    plt.savefig(f'../data/visualization_plots/{plot_counter:02d}_{feature}_synthetic.png', dpi=300)
    plt.show()
    plot_counter += 1

# ----------------- Ridge/KDE plots -----------------
# For KDE/ridge plots only
kde_features = ['reward_rate', 'win_stay', 'lose_shift']
kde_feature_labels = {
    'reward_rate': 'Reward Rate',
    'win_stay': 'Win-Stay Rate',
    'lose_shift': 'Lose-Shift Rate'
}

sns.set_style("whitegrid")
fig, axes = plt.subplots(len(kde_features), 1, figsize=(10, 8))
fig.suptitle('Distribution of Behavioral Features: Synthetic vs Real', fontsize=16, fontweight='bold')

for idx, feature in enumerate(kde_features):
    ax = axes[idx]
    for model in model_order:
        data = df[df['model_type']==model][feature]
        sns.kdeplot(data=data, ax=ax, label=f'{model} (n={len(data)})',
                    fill=True, alpha=0.5, linewidth=2, color=model_colors[model])
    ax.set_xlabel(kde_feature_labels[feature])
    ax.set_ylabel('Density')
    ax.set_title(f'{kde_feature_labels[feature]} Distribution')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'../data/visualization_plots/{plot_counter:02d}_ridge_kde_synthetic.png', dpi=300)
plt.show()

plot_counter += 1

# ----------------- Combined Subplots: Two Features per Row -----------------
ncols = 2
violin_features = [f for f in features if f not in ['choice_rate', 'win_shift', 'lose_stay']]
nrows = int(np.ceil(len(violin_features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
fig.suptitle('Behavioral Features by Model (Violinplots, Selected Features)', fontsize=18, fontweight='bold')

# Features to include in the violinplot subplots
violin_features = [f for f in features if f not in ['choice_rate', 'win_shift', 'lose_stay']]

for idx, feature in enumerate(violin_features):
    row, col = divmod(idx, ncols)
    ax = axes[row, col] if nrows > 1 else axes[col]
    present_models = [m for m in model_order if m in df['model_type'].unique()]
    present_colors = [model_colors[m] for m in present_models]
    sns.violinplot(data=df, x='model_type', y=feature, order=present_models, palette=present_colors,
                   cut=0, inner='quartile', linewidth=1, ax=ax)
    # Overlay points
    for i, model in enumerate(present_models):
        group_data = df[df['model_type'] == model][feature]
        x_positions = np.random.normal(i, 0.08, size=len(group_data))
        ax.scatter(x_positions, group_data, color=model_colors[model], edgecolor='white', s=18, alpha=0.7)
    ax.set_ylabel(feature_labels[feature], fontsize=11)
    ax.set_title(feature_labels[feature], fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    if row == nrows - 1:
        ax.set_xticks(range(len(present_models)))
    else:
        ax.set_xticklabels([])

# Hide any unused subplots
for idx in range(len(violin_features), nrows * ncols):
    row, col = divmod(idx, ncols)
    fig.delaxes(axes[row, col] if nrows > 1 else axes[col])
ax.set_xticklabels([f"{m}\n(n={len(df[df['model_type']==m])})" for m in present_models], fontsize=10)
plt.xlabel('Model Type', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f'../data/visualization_plots/combined_boxplots_all_features_2perrow.png', dpi=300)
plt.show()