import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path='AAAAsindy_analysis_with_metrics.csv'):
    """
    Load data and extract SINDY coefficients for dimensionality reduction
    """
    df = pd.read_csv(csv_path)
    
    # Extract all SINDY coefficient columns
    sindy_cols = [col for col in df.columns if col.startswith('x_')]
    
    print(f"Found {len(sindy_cols)} SINDY coefficient columns")
    
    return df, sindy_cols

def perform_dimensionality_reduction(df, sindy_cols, output_dir):
    """
    Perform PCA and t-SNE on SINDY coefficients and visualize by age
    """
    
    # Extract SINDY coefficients and age data
    sindy_data = df[sindy_cols].fillna(0)  # Fill NaN with 0
    age_data = df['Age'].dropna()
    
    # Align data (remove rows where age is missing)
    valid_indices = df['Age'].notna()
    sindy_data_clean = sindy_data[valid_indices]
    age_data_clean = df.loc[valid_indices, 'Age']
    
    print(f"Using {len(sindy_data_clean)} participants with complete age and SINDY data")
    
    # Standardize the SINDY coefficients
    scaler = StandardScaler()
    sindy_scaled = scaler.fit_transform(sindy_data_clean)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(sindy_scaled)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(sindy_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create visualizations
    create_dimensionality_plots(pca_result, tsne_result, age_data_clean, 
                               pca.explained_variance_ratio_, output_dir, df, valid_indices)
    
    return pca_result, tsne_result, age_data_clean

def create_dimensionality_plots(pca_result, tsne_result, age_data, explained_var, output_dir, df, valid_indices):
    """
    Create comprehensive dimensionality reduction plots colored by age
    """
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SINDY Structural Differences by Age: Dimensionality Reduction Analysis', 
                 fontsize=16, y=0.95)
    
    # Define age colormap
    age_min, age_max = age_data.min(), age_data.max()
    
    # 1. PCA plot with continuous age coloring
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=age_data, cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax1.set_title('PCA: Continuous Age')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Age')
    
    # 2. t-SNE plot with continuous age coloring
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=age_data, cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('t-SNE: Continuous Age')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Age')
    
    # 3. Age categories for discrete visualization
    age_categories = pd.cut(age_data, bins=3, labels=['Young', 'Middle', 'Old'])
    category_colors = {'Young': '#C6E2FF', 'Middle': '#1E90FF', 'Old': '#003366'}
    
    ax3 = axes[0, 2]
    for category in ['Young', 'Middle', 'Old']:
        mask = age_categories == category
        if mask.sum() > 0:
            ax3.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=category_colors[category], label=f'{category} (n={mask.sum()})',
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax3.set_title('PCA: Age Categories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. t-SNE with age categories
    ax4 = axes[1, 0]
    for category in ['Young', 'Middle', 'Old']:
        mask = age_categories == category
        if mask.sum() > 0:
            ax4.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=category_colors[category], label=f'{category} (n={mask.sum()})',
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.set_title('t-SNE: Age Categories')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Age distribution histogram
    ax5 = axes[1, 1]
    ax5.hist(age_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(age_data.mean(), color='red', linestyle='--', label=f'Mean: {age_data.mean():.1f}')
    ax5.axvline(age_data.median(), color='orange', linestyle='--', label=f'Median: {age_data.median():.1f}')
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Count')
    ax5.set_title('Age Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. PCA with age correlation analysis
    ax6 = axes[1, 2]
    
    # Calculate correlation between age and PC components
    pc1_age_corr = np.corrcoef(pca_result[:, 0], age_data)[0, 1]
    pc2_age_corr = np.corrcoef(pca_result[:, 1], age_data)[0, 1]
    
    # Create a plot showing the age trend in PC space
    scatter6 = ax6.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=age_data, cmap='plasma', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add trend lines if correlations are meaningful
    if abs(pc1_age_corr) > 0.1:
        z1 = np.polyfit(pca_result[:, 0], age_data, 1)
        p1 = np.poly1d(z1)
        x_trend = np.linspace(pca_result[:, 0].min(), pca_result[:, 0].max(), 100)
        # Project trend onto PC2 space (simplified)
        ax6.plot(x_trend, np.full_like(x_trend, np.median(pca_result[:, 1])), 
                'r--', alpha=0.8, linewidth=2, label=f'PC1-Age trend (r={pc1_age_corr:.3f})')
    
    ax6.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax6.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax6.set_title(f'PCA: Age Correlations\nPC1-Age: r={pc1_age_corr:.3f}\nPC2-Age: r={pc2_age_corr:.3f}')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter6, ax=ax6, label='Age')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'sindy_age_dimensionality_reduction.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dimensionality reduction plot saved to: {os.path.join(output_dir, 'sindy_age_dimensionality_reduction.png')}")

def analyze_age_clusters(pca_result, tsne_result, age_data, output_dir):
    """
    Perform clustering analysis to identify age-related structural groups
    """
    
    # Perform K-means clustering on PCA results
    n_clusters = 3  # Young, Middle, Old
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_pca = kmeans_pca.fit_predict(pca_result)
    
    kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_tsne = kmeans_tsne.fit_predict(tsne_result)
    
    # Create clustering analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Age-Related Clustering Analysis', fontsize=16, y=0.95)
    
    # PCA clusters vs age
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=clusters_pca, cmap='Set1', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax1.scatter(kmeans_pca.cluster_centers_[:, 0], kmeans_pca.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA: K-means Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE clusters vs age
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=clusters_tsne, cmap='Set1', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax2.scatter(kmeans_tsne.cluster_centers_[:, 0], kmeans_tsne.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    ax2.set_xlabel('t-SNE Dim 1')
    ax2.set_ylabel('t-SNE Dim 2')
    ax2.set_title('t-SNE: K-means Clusters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Age distribution by clusters
    ax3 = axes[1, 0]
    cluster_ages = [age_data[clusters_pca == i] for i in range(n_clusters)]
    ax3.boxplot(cluster_ages, labels=[f'Cluster {i}' for i in range(n_clusters)])
    ax3.set_ylabel('Age')
    ax3.set_title('Age Distribution by PCA Clusters')
    ax3.grid(True, alpha=0.3)
    
    # Cluster statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate cluster statistics
    cluster_stats = []
    for i in range(n_clusters):
        cluster_ages_i = age_data[clusters_pca == i]
        cluster_stats.append([
            f'Cluster {i}',
            f'{len(cluster_ages_i)}',
            f'{cluster_ages_i.mean():.1f}',
            f'{cluster_ages_i.std():.1f}',
            f'{cluster_ages_i.min():.0f}-{cluster_ages_i.max():.0f}'
        ])
    
    # Create table
    table_data = [['Cluster', 'N', 'Mean Age', 'Std Age', 'Age Range']] + cluster_stats
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    
    ax4.set_title('Cluster Statistics', pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the clustering plot
    plt.savefig(os.path.join(output_dir, 'sindy_age_clustering_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering analysis plot saved to: {os.path.join(output_dir, 'sindy_age_clustering_analysis.png')}")
    
    return clusters_pca, clusters_tsne

def create_summary_statistics(pca_result, tsne_result, age_data, sindy_cols, output_dir):
    """
    Create summary statistics and save to CSV
    """
    
    # Calculate correlations between age and principal components
    pc1_age_corr = np.corrcoef(pca_result[:, 0], age_data)[0, 1]
    pc2_age_corr = np.corrcoef(pca_result[:, 1], age_data)[0, 1]
    
    # Create summary data
    summary_data = {
        'Analysis': ['PCA', 'PCA', 't-SNE'],
        'Component': ['PC1', 'PC2', 'Overall'],
        'Age_Correlation': [pc1_age_corr, pc2_age_corr, np.nan],
        'Description': [
            f'First principal component correlation with age',
            f'Second principal component correlation with age',
            f'2D embedding of {len(sindy_cols)} SINDY coefficients'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'dimensionality_reduction_summary.csv'), index=False)
    
    print("\nDimensionality Reduction Summary:")
    print("=" * 50)
    print(f"Number of SINDY coefficients: {len(sindy_cols)}")
    print(f"Number of participants: {len(age_data)}")
    print(f"Age range: {age_data.min():.0f} - {age_data.max():.0f}")
    print(f"PC1-Age correlation: {pc1_age_corr:.3f}")
    print(f"PC2-Age correlation: {pc2_age_corr:.3f}")

def main():
    # Set up output directory
    output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy_age_structure'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("Loading data...")
    df, sindy_cols = load_and_prepare_data()
    
    print(f"Dataset shape: {df.shape}")
    
    # Perform dimensionality reduction
    print("\nPerforming dimensionality reduction...")
    pca_result, tsne_result, age_data = perform_dimensionality_reduction(df, sindy_cols, output_dir)
    
    # Analyze age-related clusters
    print("\nAnalyzing age-related clusters...")
    clusters_pca, clusters_tsne = analyze_age_clusters(pca_result, tsne_result, age_data, output_dir)
    
    # Create summary statistics
    print("\nCreating summary statistics...")
    create_summary_statistics(pca_result, tsne_result, age_data, sindy_cols, output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()