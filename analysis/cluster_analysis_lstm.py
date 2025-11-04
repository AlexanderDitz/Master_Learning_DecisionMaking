#!/usr/bin/env python3
"""
K-means clustering on LSTM hidden states
Saves results to 'lstm_kmeans_clusters.csv' and shows a 2D t-SNE plot colored by cluster.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load LSTM hidden states
hidden_df = pd.read_csv("lstm_hidden_states.csv", index_col=0)

X = hidden_df.copy()

# Normalize hidden states
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering (choose k=3 for 3 clusters)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
hidden_df['cluster'] = kmeans.fit_predict(X_scaled)

# Save results
hidden_df.to_csv("lstm_kmeans_clusters.csv")
print("✅ Saved clustering results to lstm_kmeans_clusters.csv")

# Load diagnosis info from original_data.csv
diag_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/preprocessing/original_data.csv"), usecols=['ID', 'diag'])

# Merge diagnosis info into clustering results
hidden_with_diag = hidden_df.reset_index().merge(diag_df, left_on='participant', right_on='ID', how='left')

# Unique participant counts per cluster and diagnosis
unique_counts = hidden_with_diag.groupby(['cluster', 'diag'])['participant'].nunique().unstack(fill_value=0)
print("\nUnique participants per cluster and diagnosis:")
print(unique_counts)

# Print participant IDs for each cluster
print("\nParticipant IDs by cluster:")
for cluster_id in sorted(hidden_with_diag['cluster'].unique()):
    ids = hidden_with_diag.loc[hidden_with_diag['cluster'] == cluster_id, 'participant'].unique()
    print(f"Cluster {cluster_id}: {list(ids)}\n")

# t-SNE for visualization
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X_scaled)
hidden_df['tsne1'] = X_tsne[:,0]
hidden_df['tsne2'] = X_tsne[:,1]

# Plot
group_palette = ["#d95f02", "#1b9e77", "#377eb8"]
plt.figure(figsize=(10,7))
sns.scatterplot(
    data=hidden_df,
    x='tsne1', y='tsne2',
    hue='cluster',
    palette=group_palette,
    s=100,
    alpha=0.8,
    edgecolor='black',
    linewidth=1.0
)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('lstm_kmeans_tsne_plot.png', dpi=300)
plt.show()
print("✅ Plotted t-SNE of clusters. Figure saved as lstm_kmeans_tsne_plot.png")

# Calculate mean and std for each hidden state in each cluster
cluster_stats = hidden_df.groupby('cluster').agg(['mean', 'std'])
print(cluster_stats)

from sklearn.metrics import silhouette_score
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, silhouette score={score:.3f}")
