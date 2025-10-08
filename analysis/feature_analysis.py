"""
Comprehensive Feature/Behavioral Analysis Script
Creates seaborn feature plots from participant_features.csv across diagnoses
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Change to the script directory
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# Set pandas option
pd.set_option('future.no_silent_downcasting', True)

# Load the participant features data
features_path = '../data/features/participant_features.csv'
df = pd.read_csv(features_path)
print(f"Loaded data from {features_path}")
print(f"Data shape: {df.shape}")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Create seaborn feature plots
def create_seaborn_feature_plots(df):
    """Create comprehensive seaborn plots for all behavioral features by diagnosis."""
    
    # Create visualization_plots directory if it doesn't exist
    os.makedirs('../data/visualization_plots', exist_ok=True)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Define custom colors for each diagnosis - colorblind friendly
    diagnosis_colors = {
        'Healthy': '#2E8B57',      # Sea Green
        'Depression': '#CD5C5C',   # Indian Red  
        'Bipolar': '#4682B4'       # Steel Blue
    }
    
    # Create a color palette list in the same order as diagnosis_order
    colors_list = [diagnosis_colors['Healthy'], diagnosis_colors['Depression'], diagnosis_colors['Bipolar']]
    
    # Define features to plot (excluding participant and diagnosis columns)
    features = ['choice_rate', 'reward_rate', 'win_stay', 'win_shift', 'lose_stay', 'lose_shift']
    feature_labels = {
        'choice_rate': 'Choice Rate',
        'reward_rate': 'Reward Rate', 
        'win_stay': 'Win-Stay Rate',
        'win_shift': 'Win-Shift Rate',
        'lose_stay': 'Lose-Stay Rate',
        'lose_shift': 'Lose-Shift Rate'
    }
    
    # Order diagnoses for consistent plotting
    diagnosis_order = ['Healthy', 'Depression', 'Bipolar']
    
    print(f"\n=== Creating Seaborn Feature Plots for {len(features)} Features ===")
    
    plot_counter = 1
    
    for feature in features:
        print(f"Creating plots for {feature_labels[feature]}...")
        
        # Boxplot with individual points and custom colors
        plt.figure(figsize=(12, 8))
        
        # Create boxplot with custom colors (hide outliers to avoid duplication)
        box_plot = sns.boxplot(data=df, x='diagnosis', y=feature, order=diagnosis_order, 
                              hue='diagnosis', palette=colors_list, legend=False, showfliers=False,
                              width=0.6)
        
        # Make boxplot patches transparent
        for patch in box_plot.patches:
            patch.set_alpha(0.6)
        
        # Calculate outliers for each diagnosis group to show them differently
        strip_colors = ['#1F5F3F', '#8B0000', '#2F4F4F']  # Darker versions for strip plot
        
        for i, diagnosis in enumerate(diagnosis_order):
            # Get data for this diagnosis
            group_data = df[df['diagnosis'] == diagnosis][feature]
            
            # Calculate outlier bounds (same as boxplot: Q1-1.5*IQR, Q3+1.5*IQR)
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Separate normal points and outliers
            normal_mask = (group_data >= lower_bound) & (group_data <= upper_bound)
            outlier_mask = ~normal_mask
            
            # Get x-positions for this diagnosis group
            x_pos = i
            
            # Plot normal points as filled circles with better positioning
            if normal_mask.any():
                normal_data = group_data[normal_mask]
                x_positions = np.random.normal(x_pos, 0.08, size=len(normal_data))  # Increased jitter
                plt.scatter(x_positions, normal_data, color=strip_colors[i], alpha=0.8, s=25, 
                           edgecolor='white', linewidth=0.5, zorder=10)
            
            # Plot outliers as empty circles with same size as normal points
            if outlier_mask.any():
                outlier_data = group_data[outlier_mask]
                x_positions = np.random.normal(x_pos, 0.08, size=len(outlier_data))  # Increased jitter
                plt.scatter(x_positions, outlier_data, facecolors='none', edgecolors=strip_colors[i], 
                           alpha=0.9, s=25, linewidths=2, zorder=10)
        
        # Perform statistical significance testing
        from scipy import stats
        
        # Get data for each group
        healthy_data = df[df['diagnosis'] == 'Healthy'][feature]
        depression_data = df[df['diagnosis'] == 'Depression'][feature]
        bipolar_data = df[df['diagnosis'] == 'Bipolar'][feature]
        
        # Perform pairwise comparisons (Mann-Whitney U test for non-parametric data)
        _, p_healthy_depression = stats.mannwhitneyu(healthy_data, depression_data, alternative='two-sided')
        _, p_healthy_bipolar = stats.mannwhitneyu(healthy_data, bipolar_data, alternative='two-sided')
        _, p_depression_bipolar = stats.mannwhitneyu(depression_data, bipolar_data, alternative='two-sided')
        
        # Function to get significance stars
        def get_significance_stars(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return 'ns'
        
        # Get significance stars for each comparison
        sig_healthy_depression = get_significance_stars(p_healthy_depression)
        sig_healthy_bipolar = get_significance_stars(p_healthy_bipolar)
        sig_depression_bipolar = get_significance_stars(p_depression_bipolar)
        
        # Add significance annotations to the plot
        y_max = df[feature].max()
        y_min = df[feature].min()
        y_range = y_max - y_min
        
        # Calculate positions for significance bars
        bar_height = y_max + 0.05 * y_range
        bar_height_2 = y_max + 0.12 * y_range
        bar_height_3 = y_max + 0.19 * y_range
        
        # Add significance bars and stars - clean simple lines
        # Healthy vs Depression (positions 0 and 1)
        if sig_healthy_depression != 'ns':
            plt.plot([0, 1], [bar_height, bar_height], 'k-', linewidth=1.2)
            plt.text(0.5, bar_height + 0.015 * y_range, sig_healthy_depression, 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Depression vs Bipolar (positions 1 and 2)  
        if sig_depression_bipolar != 'ns':
            plt.plot([1, 2], [bar_height_2, bar_height_2], 'k-', linewidth=1.2)
            plt.text(1.5, bar_height_2 + 0.015 * y_range, sig_depression_bipolar, 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Healthy vs Bipolar (positions 0 and 2)
        if sig_healthy_bipolar != 'ns':
            plt.plot([0, 2], [bar_height_3, bar_height_3], 'k-', linewidth=1.2)
            plt.text(1, bar_height_3 + 0.015 * y_range, sig_healthy_bipolar, 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Adjust y-axis limits to accommodate significance bars
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.30 * y_range)
        
        # Add title and axis labels
        plt.title(f'{feature_labels[feature]} - Box Plot with Individual Points', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Diagnosis', fontsize=14, labelpad=20)
        plt.ylabel(feature_labels[feature], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Create two separate legends
        
        # Legend 1: Diagnosis colors
        color_legend_elements = [
            plt.Line2D([0], [0], color=diagnosis_colors['Healthy'], linewidth=0, 
                      marker='s', markersize=10, label='Healthy'),
            plt.Line2D([0], [0], color=diagnosis_colors['Depression'], linewidth=0, 
                      marker='s', markersize=10, label='Depression'),
            plt.Line2D([0], [0], color=diagnosis_colors['Bipolar'], linewidth=0, 
                      marker='s', markersize=10, label='Bipolar')
        ]
        
        # Legend 2: Statistical significance
        significance_legend_elements = [
            plt.Line2D([0], [0], color='white', markerfacecolor='white', marker='', 
                      markersize=0, label='Statistical Significance:'),
            plt.Line2D([0], [0], color='white', markerfacecolor='white', marker='', 
                      markersize=0, label='*** p < 0.001'),
            plt.Line2D([0], [0], color='white', markerfacecolor='white', marker='', 
                      markersize=0, label='** p < 0.01'),
            plt.Line2D([0], [0], color='white', markerfacecolor='white', marker='', 
                      markersize=0, label='* p < 0.05')
        ]
        
        # Add color legend (positioned outside plot area)
        color_legend = plt.legend(handles=color_legend_elements, 
                                bbox_to_anchor=(1.02, 1), loc='upper left', 
                                fontsize=11, frameon=True, fancybox=True, shadow=True,
                                title='Diagnosis Groups', title_fontsize=12)
        
        # Add significance legend (positioned below color legend)
        significance_legend = plt.legend(handles=significance_legend_elements, 
                                       bbox_to_anchor=(1.02, 0.65), loc='upper left', 
                                       fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Add the color legend back (matplotlib only shows the last legend by default)
        plt.gca().add_artist(color_legend)
        
        plt.tight_layout()
        plt.savefig(f'../data/visualization_plots/{plot_counter:02d}_{feature}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        plot_counter += 1
        
    print(f"\n✅ Successfully created {plot_counter-1} feature plots!")
    print(f"All plots saved to '../data/visualization_plots/'")

# Call the function to create the plots
if __name__ == "__main__":
    create_seaborn_feature_plots(df)

#%%
def create_correlation_heatmap(df):
    """Create correlation heatmap of all behavioral features."""
    
    # Select only numeric features
    numeric_features = ['reward_rate', 'win_stay', 'win_shift', 'lose_stay', 'lose_shift']
    
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                fmt='.3f')
    
    plt.title('Behavioral Features Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('data/visualization_plots/25_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Created correlation heatmap")

    def create_behavioral_phenotype_clusters(df):
        """Create behavioral phenotype clusters and visualizations."""
    
    print("\n=== Creating Behavioral Phenotype Clusters ===")
    
    # Define key features for clustering (exclude complementary pairs)
    cluster_features = ['reward_rate', 'win_stay', 'lose_shift']
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])
    
    # Perform K-means clustering (3 clusters based on diagnoses)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['behavioral_cluster'] = kmeans.fit_predict(X_scaled)
    
    # Create cluster visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Behavioral Phenotype Clusters', fontsize=20, fontweight='bold')
    
    # Win-Stay vs Lose-Shift scatter plot with clusters
    scatter = ax1.scatter(df['win_stay'], df['lose_shift'], 
                         c=df['behavioral_cluster'], cmap='viridis', 
                         alpha=0.8, s=100, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Win-Stay Rate', fontsize=14)
    ax1.set_ylabel('Lose-Shift Rate', fontsize=14)
    ax1.set_title('Decision Strategy Clusters', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Behavioral Cluster')