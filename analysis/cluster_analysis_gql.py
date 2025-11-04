#!/usr/bin/env python3
"""
K-means clustering on GQL parameters
Saves results to 'gql_kmeans_clusters.csv' and shows a 2D t-SNE plot colored by cluster.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load GQL parameters
params_df = pd.read_csv("gql_parameters.csv", index_col=0)

X = params_df.drop(columns=['cluster'], errors='ignore')

# Normalize parameters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering (choose k=3 for 3 clusters)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
params_df['cluster'] = kmeans.fit_predict(X_scaled)

# Save results
params_df.to_csv("gql_kmeans_clusters.csv")
print("✅ Saved clustering results to gql_kmeans_clusters.csv")

# Load diagnosis info from original_data.csv
diag_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/preprocessing/original_data.csv"), usecols=['ID', 'diag'])

# Merge diagnosis info into clustering results
params_with_diag = params_df.reset_index().merge(diag_df, left_on='participant', right_on='ID', how='left')

# Unique participant counts per cluster and diagnosis
unique_counts = params_with_diag.groupby(['cluster', 'diag'])['participant'].nunique().unstack(fill_value=0)
print("\nUnique participants per cluster and diagnosis:")
print(unique_counts)

# Print participant IDs for each cluster
print("\nParticipant IDs by cluster:")
for cluster_id in sorted(params_with_diag['cluster'].unique()):
    ids = params_with_diag.loc[params_with_diag['cluster'] == cluster_id, 'participant'].unique()
    print(f"Cluster {cluster_id}: {list(ids)}\n")

# t-SNE for visualization
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X_scaled)
params_df['tsne1'] = X_tsne[:,0]
params_df['tsne2'] = X_tsne[:,1]

# Plot
plt.figure(figsize=(10,7))
sns.scatterplot(
    data=params_df,
    x='tsne1', y='tsne2',
    hue='cluster',
    palette = ["#d95f02", "#1b9e77", "#377eb8"],  # dark orange, dark green, dark blue
    s=100,
    alpha=0.8,
    edgecolor='black',
    linewidth=1.0
)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('gql_kmeans_tsne_plot.png', dpi=300)
plt.show()
print("✅ Plotted t-SNE of clusters. Figure saved as gql_kmeans_tsne_plot.png")

# Calculate mean and std for each parameter in each cluster
cluster_stats = params_df.groupby('cluster').agg(['mean', 'std'])
print(cluster_stats)

from sklearn.metrics import silhouette_score

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, silhouette score={score:.3f}")