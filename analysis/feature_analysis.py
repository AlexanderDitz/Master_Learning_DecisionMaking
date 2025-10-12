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
        top = y_max
        
        # Collect significant combinations
        significant_combinations = []
        
        # Check each pairwise comparison and add to list if significant
        if sig_healthy_depression != 'ns':
            significant_combinations.append([(0, 1), sig_healthy_depression])
        if sig_depression_bipolar != 'ns':
            significant_combinations.append([(1, 2), sig_depression_bipolar])
        if sig_healthy_bipolar != 'ns':
            significant_combinations.append([(0, 2), sig_healthy_bipolar])
        
        # Draw significance bars using the new bracket style
        for i, significant_combination in enumerate(significant_combinations):
            # Columns corresponding to the datasets of interest
            x1 = significant_combination[0][0]
            x2 = significant_combination[0][1]
            # What level is this bar among the bars above the plot?
            level = len(significant_combinations) - i
            # Plot the bar
            bar_height = (y_range * 0.07 * level) + top
            bar_tips = bar_height - (y_range * 0.02)
            plt.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
            )
            # Add significance text
            sig_text = significant_combination[1]
            plt.text((x1 + x2) / 2, bar_height + 0.01 * y_range, sig_text, 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Adjust y-axis limits to accommodate significance bars
        if significant_combinations:
            max_level = len(significant_combinations)
            plt.ylim(y_min - 0.05 * y_range, y_max + (0.07 * max_level + 0.05) * y_range)
        else:
            plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # Add title and axis labels
        plt.title(f'{feature_labels[feature]}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Diagnosis', fontsize=14, labelpad=40)  # Increased labelpad for sample size annotations
        plt.ylabel(feature_labels[feature], fontsize=14)
        
        # Set x-axis tick labels with sample size annotations
        x_labels = []
        for diagnosis in diagnosis_order:
            sample_size = len(df[df['diagnosis'] == diagnosis])
            x_labels.append(f'{diagnosis}\n(n={sample_size})')
        
        plt.xticks(range(len(diagnosis_order)), x_labels, fontsize=12)
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
    plt.savefig('../data/visualization_plots/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
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
    
    # Define diagnosis colors
    diagnosis_colors = {'Healthy': '#2E8B57', 'Depression': '#CD5C5C', 'Bipolar': '#4682B4'}
    
    # Create visualization with only 2 plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Behavioral Cluster Analysis', fontsize=20, fontweight='bold')

    # Plot 1: Win-Stay vs Lose-Shift scatter plot colored by diagnosis
    for diagnosis in df['diagnosis'].unique():
        mask = df['diagnosis'] == diagnosis
        ax1.scatter(df[mask]['win_stay'], df[mask]['lose_shift'], 
                   label=diagnosis, color=diagnosis_colors[diagnosis], 
                   alpha=0.8, s=100, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Win-Stay Rate', fontsize=14)
    ax1.set_ylabel('Lose-Shift Rate', fontsize=14)
    ax1.set_title('Win-Stay vs. Lose-Shift', fontsize=16, fontweight='bold')
    ax1.legend(title='Diagnosis', fontsize=12, title_fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward Rate vs Win-Stay scatter plot colored by diagnosis
    for diagnosis in df['diagnosis'].unique():
        mask = df['diagnosis'] == diagnosis
        ax2.scatter(df[mask]['reward_rate'], df[mask]['win_stay'], 
                   label=diagnosis, color=diagnosis_colors[diagnosis], 
                   alpha=0.8, s=100, edgecolors='black', linewidth=1)
    ax2.set_xlabel('Reward Rate', fontsize=14)
    ax2.set_ylabel('Win-Stay Rate', fontsize=14)
    ax2.set_title('Reward Rate vs. Win-Stay', fontsize=16, fontweight='bold')
    ax2.legend(title='Diagnosis', fontsize=12, title_fontsize=12, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/visualization_plots/08_behavioral_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Created behavioral phenotype clusters")
    return df



def create_ridge_plot(df):
    """Create ridge/KDE plots for elegant distribution comparison."""
    
    print("\n=== Creating Ridge/KDE Plots ===")
    
    # Focus on key behavioral features
    features = ['reward_rate', 'win_stay', 'lose_shift']
    feature_labels = {
        'reward_rate': 'Reward Rate', 
        'win_stay': 'Win-Stay Rate',
        'lose_shift': 'Lose-Shift Rate'
    }
    
    diagnosis_colors = {'Healthy': '#2E8B57', 'Depression': '#CD5C5C', 'Bipolar': '#4682B4'}
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Distribution of behavioral features', fontsize=18, fontweight='bold')
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Create KDE for each diagnosis with sample size in legend
        legend_labels = []
        for i, diagnosis in enumerate(['Healthy', 'Depression', 'Bipolar']):
            data = df[df['diagnosis'] == diagnosis][feature]
            sample_size = len(data)
            
            # Create KDE
            sns.kdeplot(data=data, ax=ax, label=f'{diagnosis} (n={sample_size})', 
                       color=diagnosis_colors[diagnosis], fill=True, alpha=0.6, linewidth=2)
        
        ax.set_xlabel(feature_labels[feature], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{feature_labels[feature]} Distribution', fontsize=14, fontweight='bold')
        ax.legend(title='Diagnosis')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/visualization_plots/09_ridge_kde.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Created ridge/KDE plots")



# Call the functions to create the plots
if __name__ == "__main__":
    create_correlation_heatmap(df)
    df = create_behavioral_phenotype_clusters(df)
    
    # Add only the ridge plots
    create_ridge_plot(df)