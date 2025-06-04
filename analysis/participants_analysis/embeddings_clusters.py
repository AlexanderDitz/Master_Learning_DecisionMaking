import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import os
import argparse
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal, chi2_contingency, zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
import umap
import warnings

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 12})

def calculate_correlation_with_significance(x, y):
    """Calculate correlation with significance test."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 2:
        corr, p_value = pearsonr(x[mask], y[mask])
        if p_value < 0.001:
            sig_marker = "***"
        elif p_value < 0.01:
            sig_marker = "**"
        elif p_value < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""
        return f"r = {corr:.3f}{sig_marker}"
    return "N/A"

def main():
    parser = argparse.ArgumentParser(
        description="Clustering analysis on RNN embedding data to identify distinct participant groups."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/AAAAsindy_analysis_with_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/embedding_clustering",
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load and filter data
    df = pd.read_csv(data_path)
    df = df[df['Age'] <= 45].copy()  # Age filter

    # Identify embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    print(f"CLUSTERING BASED ON: {len(embedding_cols)} RNN embedding dimensions ONLY")

    # Define metrics for downstream analysis
    behavioral_metrics = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']
    age_metrics = ['Age']  # Continuous age
    if 'Age_Category' in df.columns and not df['Age_Category'].isna().all():
        age_metrics.append('Age_Category')  # Categorical age

    all_analysis_metrics = behavioral_metrics + age_metrics

    # Drop rows missing embeddings or analysis metrics
    complete_data = df.dropna(subset=embedding_cols + all_analysis_metrics)
    print(f"Dataset: {len(complete_data)} participants (age ≤45)")
    cont_age_count = len(complete_data.dropna(subset=['Age']))
    cat_age_count = len(complete_data.dropna(subset=['Age_Category'])) if 'Age_Category' in complete_data.columns else 0
    print(f"Available age data: Continuous={cont_age_count}, Categorical={cat_age_count}")

    # Standardize embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(complete_data[embedding_cols])
    print(f"Standardized {len(embedding_cols)} embedding features for clustering")

    # PCA preprocessing
    n_pca = min(16, len(embedding_cols))
    pca_preprocessing = PCA(n_components=n_pca)
    X_pca = pca_preprocessing.fit_transform(X_scaled)
    explained_var = pca_preprocessing.explained_variance_ratio_.sum()
    print(f"PCA preprocessing: reduced to {X_pca.shape[1]} components explaining {explained_var:.1%} variance")

    # OUTLIER DETECTION
    # Statistical outliers via Z-score
    z_scores = np.abs(stats.zscore(X_pca, axis=0))
    outlier_threshold = 3.5
    statistical_outliers = np.any(z_scores > outlier_threshold, axis=1)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_outliers = iso_forest.fit_predict(X_pca) == -1

    # Combine outlier masks
    combined_outliers = statistical_outliers | isolation_outliers
    n_outliers = np.sum(combined_outliers)
    stat_count = np.sum(statistical_outliers)
    iso_count = np.sum(isolation_outliers)
    print(f"OUTLIER DETECTION:")
    print(f"  Statistical outliers (Z > {outlier_threshold}): {stat_count}")
    print(f"  Isolation Forest outliers: {iso_count}")
    print(f"  Combined outliers: {n_outliers} ({n_outliers/len(complete_data)*100:.1f}%)")

    # Create clean dataset without outliers for clustering
    clean_mask = ~combined_outliers
    complete_data_clean = complete_data[clean_mask].copy()
    X_clean = X_pca[clean_mask]
    print(f"Clean dataset for clustering: {len(complete_data_clean)} participants (removed {n_outliers} outliers)")

    # Prepare X_for_clustering
    X_for_clustering = X_clean

    # STEP 2: DIMENSIONALITY REDUCTION FOR VISUALIZATION
    pca_vis = PCA(n_components=2)
    pca_result_all = pca_vis.fit_transform(X_scaled)  # All data including outliers

    perplexity = min(30, max(5, len(complete_data_clean)//4))
    tsne_vis = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result_all = tsne_vis.fit_transform(X_scaled)

    n_neighbors = min(15, max(3, len(complete_data_clean)//5))
    umap_vis = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    umap_result_all = umap_vis.fit_transform(X_scaled)

    # STEP 3: CLUSTER OPTIMIZATION (focus on k=4)
    max_clusters = min(6, len(complete_data_clean) - 1)
    cluster_range = range(2, max_clusters + 1)
    silhouette_scores = []
    ch_scores = []
    wcss_scores = []
    balance_scores = []

    print(f"\nTesting {len(cluster_range)} different cluster numbers on clean data...")

    for k in cluster_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels_temp = kmeans_temp.fit_predict(X_for_clustering)

        sil_score = silhouette_score(X_for_clustering, labels_temp)
        ch_score = calinski_harabasz_score(X_for_clustering, labels_temp)
        wcss = kmeans_temp.inertia_

        cluster_counts = np.bincount(labels_temp)
        max_cluster_prop = np.max(cluster_counts) / len(labels_temp)
        balance_score = 1 - max_cluster_prop
        min_cluster_size = np.min(cluster_counts)
        min_cluster_prop = min_cluster_size / len(labels_temp)

        silhouette_scores.append(sil_score)
        ch_scores.append(ch_score)
        wcss_scores.append(wcss)
        balance_scores.append(balance_score)

        print(f"k={k}: Silhouette={sil_score:.3f}, CH={ch_score:.1f}, WCSS={wcss:.1f}, "
              f"Balance={balance_score:.3f}, MinSize={min_cluster_size} ({min_cluster_prop:.1%})")

    # STEP 4: SET OPTIMAL CLUSTERS TO 4
    optimal_k = 4
    print(f"\nSELECTED OPTIMAL: {optimal_k} clusters (based on analysis)")
    k4_idx = optimal_k - 2
    print(f"k=4 metrics: Silhouette={silhouette_scores[k4_idx]:.3f}, Balance={balance_scores[k4_idx]:.3f}")

    # STEP 5: FINAL CLUSTERING ON PCA-PREPROCESSED EMBEDDINGS (WITHOUT OUTLIERS)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels_clean = kmeans.fit_predict(X_for_clustering)
    complete_data_clean = complete_data_clean.copy()
    complete_data_clean['cluster'] = cluster_labels_clean

    # Handle outliers
    if n_outliers > 0:
        outlier_data = complete_data[combined_outliers].copy()
        X_outliers = X_pca[combined_outliers]
        outlier_clusters = kmeans.predict(X_outliers)
        outlier_data['cluster'] = outlier_clusters
        outlier_data['is_outlier'] = True

        complete_data_clean['is_outlier'] = False
        complete_data_final = pd.concat([complete_data_clean, outlier_data], ignore_index=True)
    else:
        complete_data_final = complete_data_clean.copy()
        complete_data_final['is_outlier'] = False

    cluster_labels_final = complete_data_final['cluster'].values

    print(f"\nFINAL CLUSTERS (outliers handled):")
    cluster_sizes = []
    for i in range(optimal_k):
        count = np.sum(cluster_labels_final == i)
        percentage = (count / len(cluster_labels_final)) * 100
        outliers_in_cluster = np.sum((cluster_labels_final == i) & (complete_data_final['is_outlier']))
        cluster_sizes.append(count)
        print(f"Cluster {i}: {count} participants ({percentage:.1f}%) [outliers: {outliers_in_cluster}]")

    max_cluster_percentage = max(cluster_sizes) / len(cluster_labels_final) * 100
    print(f"\nClustering appears balanced (largest cluster: {max_cluster_percentage:.1f}%)")
    if n_outliers > 0:
        print(f"Note: {n_outliers} outliers were detected and assigned to nearest clusters")

    # STEP 6: COMPREHENSIVE CORRELATION ANALYSIS
    print(f"\nAnalyzing how RNN embedding clusters relate to behavioral and age variables...")
    complete_data = complete_data_final

    # Embedding-behavior correlations
    correlations = []
    for metric in all_analysis_metrics:
        if metric != 'Age_Category':  # Skip categorical for correlation
            for col in embedding_cols:
                r, p = pearsonr(complete_data[col], complete_data[metric])
                correlations.append({
                    'embedding': col,
                    'behavioral_metric': metric,
                    'correlation': r,
                    'p_value': p,
                    'abs_corr': abs(r)
                })

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    # VISUALIZATIONS

    # 1. Comprehensive cluster optimization with balance consideration
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cluster Optimization with Balance Consideration (RNN Embeddings)', fontsize=16)

    axes[0, 0].plot(cluster_range, silhouette_scores, 'o-', color='blue')
    axes[0, 0].axvline(optimal_k, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Selected k={optimal_k}')
    axes[0, 0].set_title('Silhouette Score (higher better)')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(cluster_range, ch_scores, 'o-', color='orange')
    axes[0, 1].axvline(optimal_k, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Selected k={optimal_k}')
    axes[0, 1].set_title('Calinski-Harabasz Score (higher better)')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('CH Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(cluster_range, wcss_scores, 'o-', color='purple')
    axes[0, 2].axvline(optimal_k, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Selected k={optimal_k}')
    axes[0, 2].set_title('WCSS - Elbow Method (lower better)')
    axes[0, 2].set_xlabel('Number of Clusters')
    axes[0, 2].set_ylabel('WCSS')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(cluster_range, balance_scores, 'o-', color='green')
    axes[1, 0].axvline(optimal_k, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Selected k={optimal_k}')
    axes[1, 0].set_title('Cluster Balance Score (higher better)')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Balance Score (1 - max_proportion)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    balance_weighted_silhouette = np.array(silhouette_scores) * np.array(balance_scores)
    axes[1, 1].plot(cluster_range, balance_weighted_silhouette, 'o-', color='red')
    axes[1, 1].axvline(optimal_k, color='red', linestyle='-', linewidth=3, alpha=0.8, label=f'Selected k={optimal_k}')
    axes[1, 1].set_title('Balance-Weighted Silhouette')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Silhouette × Balance')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    k4_labels = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(X_for_clustering)
    k4_sizes = np.bincount(k4_labels)
    colors_bar = plt.cm.viridis(np.linspace(0, 1, 4))
    axes[1, 2].bar(range(4), k4_sizes, color=colors_bar, alpha=0.7)
    axes[1, 2].set_title(f'Cluster Sizes for k={optimal_k}')
    axes[1, 2].set_xlabel('Cluster ID')
    axes[1, 2].set_ylabel('Number of Participants')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_comprehensive_cluster_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Dimensionality reduction with optimal clusters
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = cluster_labels_final.copy().astype(float)
    outlier_mask = complete_data['is_outlier']

    # PCA
    scatter1 = axes[0].scatter(pca_result_all[:, 0], pca_result_all[:, 1],
                               c=colors, cmap='viridis', s=80, alpha=0.8)
    if np.any(outlier_mask):
        axes[0].scatter(
            pca_result_all[outlier_mask, 0],
            pca_result_all[outlier_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=100,
            linewidth=2
        )
    axes[0].set_xlabel(f'PCA 1 ({pca_vis.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PCA 2 ({pca_vis.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title(f'PCA: {optimal_k} RNN Embedding Clusters')
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    scatter2 = axes[1].scatter(tsne_result_all[:, 0], tsne_result_all[:, 1],
                               c=colors, cmap='viridis', s=80, alpha=0.8)
    if np.any(outlier_mask):
        axes[1].scatter(
            tsne_result_all[outlier_mask, 0],
            tsne_result_all[outlier_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=100,
            linewidth=2
        )
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title(f't-SNE: {optimal_k} RNN Embedding Clusters')
    axes[1].grid(True, alpha=0.3)

    # UMAP
    scatter3 = axes[2].scatter(umap_result_all[:, 0], umap_result_all[:, 1],
                               c=colors, cmap='viridis', s=80, alpha=0.8)
    if np.any(outlier_mask):
        axes[2].scatter(
            umap_result_all[outlier_mask, 0],
            umap_result_all[outlier_mask, 1],
            facecolors='none',
            edgecolors='red',
            s=100,
            linewidth=2
        )
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].set_title(f'UMAP: {optimal_k} RNN Embedding Clusters')
    axes[2].grid(True, alpha=0.3)

    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    plt.colorbar(scatter3, cax=cbar_ax, label='Cluster')
    if np.any(outlier_mask):
        cbar_ax.text(1.5, -0.1, 'Red edges = outliers', transform=cbar_ax.transAxes,
                     rotation=90, fontsize=10, ha='center')

    plt.savefig(os.path.join(output_dir, '2_embedding_clusters_dimensionality_reduction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Behavioral + Age differences by cluster
    analysis_metrics_for_plot = [m for m in all_analysis_metrics if m != 'Age_Category']
    n_cols = 3
    n_rows = (len(analysis_metrics_for_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    fig.suptitle(f'How Do RNN Embedding Clusters Differ? (Age ≤45)', fontsize=16)

    axes = axes.flatten() if n_rows > 1 else axes

    for i, metric in enumerate(analysis_metrics_for_plot):
        if i < len(axes):
            ax = axes[i]
            sns.boxplot(x='cluster', y=metric, data=complete_data, palette='viridis', ax=ax)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('RNN Embedding Cluster')

            groups = [
                complete_data[complete_data['cluster'] == c][metric].dropna()
                for c in range(optimal_k)
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                _, p_value = f_oneway(*groups)
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                p_text = f"p={'<0.001' if p_value < 0.001 else f'={p_value:.3f}'}{significance}"
                ax.text(
                    0.05, 0.95, p_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    for i in range(len(analysis_metrics_for_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '3_behavioral_age_differences_by_cluster.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Age category analysis (if available)
    if 'Age_Category' in complete_data.columns and not complete_data['Age_Category'].isna().all():
        contingency_table = pd.crosstab(complete_data['cluster'], complete_data['Age_Category'])
        chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Stacked bar chart
        contingency_table.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
        axes[0].set_title(f'Age Categories by RNN Embedding Cluster\nChi-square test: p={p_chi2:.3f}')
        axes[0].set_xlabel('RNN Embedding Cluster')
        axes[0].set_ylabel('Number of Participants')
        axes[0].legend(title='Age Category', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Proportion heatmap
        contingency_prop = contingency_table.div(contingency_table.sum(axis=1), axis=0)
        sns.heatmap(contingency_prop, annot=True, fmt='.2f', cmap='viridis', ax=axes[1])
        axes[1].set_title('Proportion of Age Categories within Each Cluster')
        axes[1].set_xlabel('Age Category')
        axes[1].set_ylabel('RNN Embedding Cluster')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_age_category_by_cluster.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nAge category distribution across clusters (Chi-square p={p_chi2:.3f}):")
        print(contingency_table)

    # 5. Embedding-behavior correlation heatmap
    top_embeddings = corr_df.sort_values('abs_corr', ascending=False)['embedding'].unique()[:15]
    pivot_df = corr_df.pivot(index='behavioral_metric', columns='embedding', values='correlation')
    pivot_subset = pivot_df[top_embeddings]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot_subset,
        cmap='RdBu_r',
        annot=True,
        fmt='.2f',
        center=0,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    plt.title('Top RNN Embedding-Behavior/Age Correlations')
    plt.xlabel('RNN Embedding Dimensions')
    plt.ylabel('Behavioral/Age Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_embedding_behavior_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Which variables best explain the clusters?
    cluster_feature_corrs = []
    for feature in analysis_metrics_for_plot:
        rho, p = spearmanr(complete_data['cluster'], complete_data[feature])
        cluster_feature_corrs.append({'Feature': feature, 'Correlation': rho, 'p-value': p, 'abs_corr': abs(rho)})

    feat_corr_df = pd.DataFrame(cluster_feature_corrs).sort_values('abs_corr', ascending=False)

    plt.figure(figsize=(12, 8))
    colors = ['blue' if x >= 0 else 'red' for x in feat_corr_df['Correlation']]
    bars = plt.barh(feat_corr_df['Feature'], feat_corr_df['Correlation'], color=colors)

    # Add significance markers
    for i, (p, corr) in enumerate(zip(feat_corr_df['p-value'], feat_corr_df['Correlation'])):
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = ''
        if marker:
            x_pos = corr + (0.05 if corr >= 0 else -0.05)
            plt.text(x_pos, i, marker, ha='center', va='center', fontsize=12, fontweight='bold')

    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
    plt.xlabel('Spearman Correlation with RNN Embedding Cluster')
    plt.title('Which Variables Best Explain RNN Embedding Clusters?\n(*p<0.05, **p<0.01, ***p<0.001)')
    plt.grid(True, axis='x', alpha=0.3)

    strongest_feature = feat_corr_df.iloc[0]
    plt.text(
        0.02, 0.98,
        f"Strongest predictor: {strongest_feature['Feature']}\n"
        f"(r={strongest_feature['Correlation']:.3f}, p={strongest_feature['p-value']:.3f})",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        verticalalignment='top'
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_what_explains_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Cluster profiles radar chart
    radar_metrics = analysis_metrics_for_plot
    cluster_radar_data = []
    for cluster_id in range(optimal_k):
        cluster_data = complete_data[complete_data['cluster'] == cluster_id]
        cluster_values = []
        for metric in radar_metrics:
            metric_mean = cluster_data[metric].mean()
            metric_std = complete_data[metric].std()
            metric_mean_overall = complete_data[metric].mean()
            z_score = (metric_mean - metric_mean_overall) / metric_std if metric_std > 0 else 0
            cluster_values.append(z_score)
        cluster_radar_data.append(cluster_values)

    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
    for i, values in enumerate(cluster_radar_data):
        vals = values + values[:1]
        n_participants = len(complete_data[complete_data['cluster'] == i])
        ax.plot(angles, vals, linewidth=3, label=f'Cluster {i} (n={n_participants})', color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
    ax.set_title('RNN Embedding Cluster Profiles\n(Z-scores relative to population)', size=15, pad=30)
    ax.grid(True)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_cluster_profiles_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. PCA overlay with behavioral metrics
    n_cols = 3
    n_rows = (len(analysis_metrics_for_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15))
    fig.suptitle('Behavioral/Age Variables Mapped onto RNN Embedding Space', fontsize=16)
    axes = axes.flatten() if n_rows > 1 else axes

    for i, metric in enumerate(analysis_metrics_for_plot):
        if i < len(axes):
            ax = axes[i]
            scatter = ax.scatter(
                pca_result_all[:, 0],
                pca_result_all[:, 1],
                c=complete_data[metric],
                cmap='coolwarm',
                s=60,
                alpha=0.8
            )

            for cluster_idx in range(optimal_k):
                cluster_points = pca_result_all[complete_data['cluster'] == cluster_idx]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    ax.text(
                        center[0],
                        center[1],
                        str(cluster_idx),
                        fontsize=14,
                        ha='center',
                        va='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8, edgecolor='black')
                    )

            ax.set_xlabel(f'PCA 1 ({pca_vis.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PCA 2 ({pca_vis.explained_variance_ratio_[1]:.1%})')
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(metric.replace('_', ' ').title())

    for i in range(len(analysis_metrics_for_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '8_pca_behavioral_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Hierarchical clustering dendrogram
    Z = linkage(X_for_clustering, method='ward')
    plt.figure(figsize=(15, 8))

    if len(complete_data_clean) > optimal_k * 3:
        dendrogram(Z, truncate_mode='lastp', p=int(optimal_k * 3),
                   leaf_rotation=90, leaf_font_size=10, show_contracted=True)
    else:
        dendrogram(Z, leaf_rotation=90, leaf_font_size=8)

    plt.title('Hierarchical Clustering of RNN Embeddings\n(Ward linkage method)')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9_hierarchical_clustering.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary CSVs
    cluster_summary = complete_data.groupby('cluster').agg({
        'Age': ['mean', 'std', 'count'],
        **{metric: ['mean', 'std'] for metric in behavioral_metrics}
    })
    cluster_summary.to_csv(os.path.join(output_dir, 'cluster_summary_statistics.csv'), index=True)

    pd.DataFrame(cluster_feature_corrs).to_csv(
        os.path.join(output_dir, 'what_explains_clusters.csv'), index=False
    )
    corr_df.to_csv(os.path.join(output_dir, 'embedding_behavior_correlations.csv'), index=False)

    # Print summary to console
    significant_corrs = corr_df[corr_df['p_value'] < 0.05]
    strong_corrs = significant_corrs[significant_corrs['abs_corr'] > 0.3]

    print("\n" + "="*70)
    print(" RNN EMBEDDING CLUSTERING ANALYSIS RESULTS")
    print("="*70)
    print("CLUSTERING METHOD: K-means on PCA-preprocessed RNN embedding dimensions")
    print(f"PREPROCESSING: {len(embedding_cols)} → {X_for_clustering.shape[1]} dimensions via PCA")
    print(f"OPTIMAL CLUSTERS: {optimal_k} clusters (from {len(complete_data)} participants, age ≤45)")
    print(f"PCA VARIANCE EXPLAINED: {sum(pca_vis.explained_variance_ratio_):.1%}")
    print(f"SILHOUETTE SCORE: {silhouette_scores[k4_idx]:.3f}")
    print(f"BALANCE SCORE: {balance_scores[k4_idx]:.3f}")

    print("\nCLUSTER SIZES:")
    for i in range(optimal_k):
        count = np.sum(cluster_labels_final == i)
        percentage = (count / len(cluster_labels_final)) * 100
        outliers_in_cluster = np.sum((cluster_labels_final == i) & (complete_data['is_outlier']))
        print(f"  Cluster {i}: {count} participants ({percentage:.1f}%) [outliers: {outliers_in_cluster}]")

    print("\nEMBEDDING-BEHAVIOR RELATIONSHIPS:")
    print(f"  {len(significant_corrs)}/{len(corr_df)} significant correlations (p<0.05)")
    print(f"  {len(strong_corrs)} strong correlations (|r|>0.3)")
    if len(strong_corrs) > 0:
        strongest_corr = corr_df.iloc[0]
        print(f"  Strongest: {strongest_corr['embedding']} ↔ {strongest_corr['behavioral_metric']} (r={strongest_corr['correlation']:.3f})")

    strongest_predictor = feat_corr_df.iloc[0]
    print(f"\nBEST CLUSTER PREDICTOR: {strongest_predictor['Feature']} (r={strongest_predictor['Correlation']:.3f}, p={strongest_predictor['p-value']:.3f})")

    print("\nOUTLIER ANALYSIS:")
    if n_outliers > 0:
        print(f"  {n_outliers} outliers detected ({n_outliers/len(complete_data)*100:.1f}% of data)")
        outlier_distribution = dict(complete_data[complete_data['is_outlier']]['cluster'].value_counts().sort_index())
        print(f"  Outliers distributed across clusters: {outlier_distribution}")
    else:
        print("  No outliers detected")

if __name__ == "__main__":
    main()
