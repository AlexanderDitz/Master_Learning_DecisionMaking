import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ----------------- Setup -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
pd.set_option('future.no_silent_downcasting', True)

# ----------------- Configuration -----------------
model_order = ['HUMAN', 'LSTM', 'BENCHMARK', 'RNN', 'SPICE']
model_colors = {'HUMAN': '#2E8B57', 'LSTM': '#FF4500', 'BENCHMARK': '#FF6347', 'RNN': '#7B68EE', 'SPICE': '#4682B4'}
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

def load_all_data():
    # Load all datasets and add model_type column
    rnn_df = pd.read_csv('../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_rnn_l2_0_001.csv')
    rnn_df['model_type'] = 'RNN'
    spice2_df = pd.read_csv('../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice2_l2_0_001.csv')
    spice2_df['model_type'] = 'SPICE'
    benchmark_df = pd.read_csv('../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_benchmark.csv')
    benchmark_df['model_type'] = 'BENCHMARK'
    lstm_df = pd.read_csv('../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_lstm.csv')
    lstm_df['model_type'] = 'LSTM'
    real_df = pd.read_csv('../data/features/real_features/real_participant_features.csv')
    real_df['model_type'] = 'HUMAN'
    # Combine
    df = pd.concat([real_df, rnn_df, spice2_df, benchmark_df, lstm_df], ignore_index=True)
    return df

def plot_boxplots_with_points(df, features, feature_labels, model_order, model_colors, plot_counter=1):
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
    return plot_counter

def plot_ridge_kde(df, kde_features, kde_feature_labels, model_order, model_colors, plot_counter=1):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(len(kde_features), 1, figsize=(10, 8))
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
    return plot_counter + 1

def plot_combined_subplots(df, features, feature_labels, model_order, model_colors, plot_counter=1, ncols=2):
    violin_features = features
    nrows = int(np.ceil(len(violin_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
    for idx, feature in enumerate(violin_features):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        present_models = [m for m in model_order if m in df['model_type'].unique()]
        present_colors = [model_colors[m] for m in present_models]
        sns.boxplot(data=df, x='model_type', y=feature, order=present_models, palette=present_colors,
                    showfliers=False, width=0.6, ax=ax)
        # Overlay points
        for i, model in enumerate(present_models):
            group_data = df[df['model_type'] == model][feature]
            x_positions = np.random.normal(i, 0.08, size=len(group_data))
            ax.scatter(x_positions, group_data, color=model_colors[model], edgecolor='white', s=18, alpha=0.7)
        ax.set_ylabel(feature_labels[feature], fontsize=11)
        ax.set_title(feature_labels[feature], fontsize=13)
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
    plt.savefig(f'../data/visualization_plots/{plot_counter:02d}_combined_feature_boxplot_real_synthetic.png', dpi=300)
    plt.show()
    return plot_counter + 1

if __name__ == "__main__":
    plot_counter = 1
    df = load_all_data()
    # Boxplots with points for all features
    plot_counter = plot_boxplots_with_points(df, features, feature_labels, model_order, model_colors, plot_counter)
    # Ridge/KDE plots for selected features
    kde_features = ['reward_rate', 'choice_perseveration', 'switch_rate']
    kde_feature_labels = {
        'reward_rate': 'Reward Rate',
        'choice_perseveration': 'Choice Perseveration',
        'switch_rate': 'Switch Rate'
    }
    plot_counter = plot_ridge_kde(df, kde_features, kde_feature_labels, model_order, model_colors, plot_counter)
    # Combined subplots for a subset of features (customize as needed)
    combined_features = ['choice_rate', 'reward_rate', 'win_stay', 'lose_shift', 'choice_perseveration', 'switch_rate']
    plot_counter = plot_combined_subplots(df, combined_features, feature_labels, model_order, model_colors, plot_counter)