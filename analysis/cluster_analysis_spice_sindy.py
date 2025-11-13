import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Load wide-format SINDy coefficients (with lists as strings)
df = pd.read_csv("results/model_params/spice_sindy_parameters.csv")
print(f"Loaded SINDy parameters: {df.shape[0]} participants, {df.shape[1]} features (including participant column)")

# Expand all list columns into separate columns
feature_cols = [col for col in df.columns if col != 'participant']
expanded = []
for col in feature_cols:
    # Check if column contains lists as strings
    if df[col].apply(lambda x: isinstance(x, str) and x.startswith('[')).any():
        # Expand this column
        expanded_arrays = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x])
        maxlen = expanded_arrays.apply(len).max()
        for i in range(maxlen):
            df[f"{col}_{i}"] = expanded_arrays.apply(lambda arr: arr[i] if i < len(arr) else 0)
        df = df.drop(columns=[col])

# Merge diagnosis info
meta = pd.read_csv("data/preprocessing/original_data.csv", usecols=["ID", "diag"])
meta = meta.drop_duplicates(subset="ID").rename(columns={"ID": "participant", "diag": "diagnosis"})
df = df.merge(meta, on="participant", how="left")

# Prepare data for clustering (drop non-feature columns and keep only numeric columns)
feature_cols = [col for col in df.columns if col not in ['participant', 'diagnosis'] and np.issubdtype(df[col].dtype, np.number)]
X = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
# k = 3
k = 6
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['cluster'] = labels

# t-SNE for visualization
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X_scaled)
df['tsne1'] = X_tsne[:,0]
df['tsne2'] = X_tsne[:,1]

plt.figure(figsize=(10,7))
palette = [
    "#d95f02",  # dark orange
    "#1b9e77",  # dark green
    "#377eb8",  # dark blue
    "#ffd700",  # bright yellow (Gold)
    "#e41a1c",  # bright red
    "#e7298a"   # magenta/pink
]
# palette = ["#d95f02", "#1b9e77", "#377eb8"]  # dark orange, dark green, dark blue
sns.scatterplot(
    data=df,
    x='tsne1', y='tsne2',
    hue='cluster',
    palette=palette,
    s=100,
    alpha=0.8,
    edgecolor='black',
    linewidth=1.0
)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/clustering_analysis/spice_sindy_tsne_plot.png', dpi=300)
plt.show()
print("✅ Plotted t-SNE of clusters. Figure saved as results/clustering_analysis/spice_sindy_tsne_plot.png")

# Silhouette scores for k=2 to 10
print("\nSilhouette scores for k=2 to 10:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(X_scaled)
    score_k = silhouette_score(X_scaled, labels_k)
    print(f"k={k}, silhouette score={score_k:.3f}")

# Unique participant counts per cluster and diagnosis
unique_counts = df.groupby(['cluster', 'diagnosis'])['participant'].nunique().unstack(fill_value=0)
print("\nUnique participants per cluster and diagnosis:")
print(unique_counts)

# Mean and std for each parameter in each cluster
cluster_stats = df.groupby('cluster')[feature_cols].agg(['mean', 'std'])
print("\nMean and std for each parameter in each cluster:")
print(cluster_stats)
# Save to CSV for further inspection
cluster_stats.to_csv("results/clustering_analysis/spice_sindy_cluster_stats_mean_std.csv")
print("✅ Saved cluster mean/std table to spice_sindy_cluster_stats_mean_std.csv")

# Print participant IDs for each cluster
print("\nParticipant IDs by cluster:")
for cluster_id in sorted(df['cluster'].unique()):
    ids = df.loc[df['cluster'] == cluster_id, 'participant'].unique()
    print(f"Cluster {cluster_id}: {list(ids)}\n")

# Save cluster assignments
df[['participant', 'cluster', 'diagnosis']].to_csv("results/clustering_analysis/spice_sindy_clusters.csv", index=False)
print("\n✅ Clustering complete. Results saved to spice_sindy_clusters.csv and spice_sindy_tsne_plot.png")
