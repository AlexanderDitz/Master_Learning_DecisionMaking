"""
SPICE_vs_LSTM: Clustering and t-SNE visualization of synthetic participants from SPICE and LSTM/Q agent models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import os

# Get absolute path to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute paths to synthetic feature CSVs
spice_path = os.path.join(script_dir, '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_spice2_l2_0_001.csv')
lstm_path = os.path.join(script_dir, '../data/features/synthetic_features/synthetic_features_dezfouli2019_generated_behavior_lstm.csv')

# Load data
spice_df = pd.read_csv(spice_path)
lstm_df = pd.read_csv(lstm_path)

# Combine data
all_df = pd.concat([spice_df, lstm_df], ignore_index=True)

# Features to use for clustering/t-SNE
feature_cols = [
    'choice_rate', 'reward_rate', 'win_stay', 'win_shift',
    'lose_stay', 'lose_shift', 'choice_perseveration', 'switch_rate'
]

# Drop rows with missing values in features
all_df = all_df.dropna(subset=feature_cols)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_df[feature_cols])

# KMeans clustering (k=2 for model separation, or set k=3+ for more clusters)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
all_df['cluster'] = kmeans.fit_predict(X_scaled)

# t-SNE embedding
print('Running t-SNE...')
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X_scaled)
all_df['tsne1'] = X_tsne[:, 0]
all_df['tsne2'] = X_tsne[:, 1]

# Visualization
def plot_tsne(df, color_by='model_type', save_path=None):
    plt.figure(figsize=(10, 7))
    palette = {'spice2': '#d95f02', 'lstm': '#1b9e77'}  # dark orange, dark green
    sns.scatterplot(
        data=df,
        x='tsne1', y='tsne2',
        hue=color_by,
        palette=palette if color_by == 'model_type' else 'tab10',
        marker='o',
        s=100, alpha=0.85, edgecolor='black', linewidth=1.0
    )
    plt.title(f't-SNE of Synthetic Participants by {color_by}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Plot by model type only
plot_tsne(all_df, color_by='model_type', save_path=os.path.join(script_dir, 'tsne_by_model_type.png'))

print('✅ t-SNE clustering and plot complete. Figure saved to the project root folder.')

# --- Sequence Clustering Section ---
# Absolute paths to synthetic data (raw sequences)
spice_seq_path = os.path.join(script_dir, '../data/synthetic_data/dezfouli2019_generated_behavior_spice2_l2_0_001.csv')
lstm_seq_path = os.path.join(script_dir, '../data/synthetic_data/dezfouli2019_generated_behavior_lstm.csv')

# Load sequence data
spice_seq_df = pd.read_csv(spice_seq_path)
lstm_seq_df = pd.read_csv(lstm_seq_path)

# Assumes both CSVs have a 'choice' column with comma-separated string of 0/1 (or L/R)
def extract_choice_sequences(df, col='choice'):
    # Try to convert to list of ints (0/1)
    return df[col].apply(lambda x: [int(i) for i in str(x).replace('[','').replace(']','').replace(' ','').split(',') if i != ''])

# Only keep participants with valid sequences in both models and equal length
if 'choice' in spice_seq_df.columns and 'choice' in lstm_seq_df.columns:
    spice_seq_df['seq'] = extract_choice_sequences(spice_seq_df)
    lstm_seq_df['seq'] = extract_choice_sequences(lstm_seq_df)
    all_seq = pd.concat([
        spice_seq_df[['id','model_type','seq']],
        lstm_seq_df[['id','model_type','seq']]
    ], ignore_index=True)
    # Filter to equal-length sequences
    seq_lens = all_seq['seq'].apply(len)
    mode_len = seq_lens.mode()[0]
    all_seq = all_seq[seq_lens == mode_len].reset_index(drop=True)
    seq_matrix = np.vstack(all_seq['seq'].to_list())
    # Compute Hamming distance matrix
    dist_matrix = pairwise_distances(seq_matrix, metric='hamming')
    # Agglomerative clustering
    n_clusters = 2
    agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    all_seq['seq_cluster'] = agg.fit_predict(dist_matrix)
    # t-SNE on distance matrix
    tsne_seq = TSNE(n_components=2, metric='precomputed', random_state=42, perplexity=20)
    X_seq_tsne = tsne_seq.fit_transform(dist_matrix)
    all_seq['tsne1'] = X_seq_tsne[:,0]
    all_seq['tsne2'] = X_seq_tsne[:,1]
    # Plot
    plt.figure(figsize=(10,7))
    palette = {'spice2': '#d95f02', 'lstm': '#1b9e77'}
    sns.scatterplot(
        data=all_seq,
        x='tsne1', y='tsne2',
        hue='model_type',
        palette=palette,
        marker='o',
        s=100, alpha=0.85, edgecolor='black', linewidth=1.0
    )
    plt.title('t-SNE of Synthetic Participants by Sequence (Agglomerative Clustering)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='model_type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'tsne_by_sequence.png'), dpi=300)
    plt.show()
    print('✅ Sequence clustering and plot complete. Figure saved to the project root folder.')
else:
    print('⚠️  No sequence data found in both files. Skipping sequence clustering.')
