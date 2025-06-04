import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# SINDy coefficient columns
SINDY_COEFFS = [
    'x_learning_rate_reward_c_value_reward',
    'x_value_reward_not_chosen_x_value_reward_not_chosen', 
    'x_value_reward_not_chosen_c_value_choice',
    'x_learning_rate_reward_x_learning_rate_reward',
    'x_learning_rate_reward_1',
    'x_value_reward_not_chosen_c_reward_chosen',
    'x_learning_rate_reward_c_value_choice',
    'x_value_choice_not_chosen_1',
    'x_value_choice_chosen_c_value_reward',
    'x_value_choice_chosen_1',
    'x_value_choice_not_chosen_c_value_reward',
    'x_learning_rate_reward_c_reward_chosen',
    'x_value_choice_chosen_x_value_choice_chosen',
    'x_value_reward_not_chosen_1',
    'x_value_choice_not_chosen_x_value_choice_not_chosen',
    'params_x_learning_rate_reward',
    'params_x_value_reward_not_chosen',
    'params_x_value_choice_chosen',
    'params_x_value_choice_not_chosen'
]

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    return df

def plot_coefficient_presence_by_age(df, out_dir):
    """Plot presence/absence of SINDy coefficients by age groups"""
    
    # Create binary presence matrix
    presence_matrix = df[SINDY_COEFFS].notna().astype(int)
    presence_matrix['Age_Category'] = df['Age_Category']
    presence_matrix['Age'] = df['Age']
    
    # Coefficient presence rates by age category
    if 'Age_Category' in df.columns:
        presence_by_age_cat = presence_matrix.groupby('Age_Category')[SINDY_COEFFS].mean()
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(presence_by_age_cat.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Presence Rate'})
        plt.title('SINDy Coefficient Presence Rate by Age Category')
        plt.xlabel('Age Category')
        plt.ylabel('SINDy Coefficients')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / 'coeff_presence_by_age_category.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Overall presence rates
    presence_rates = presence_matrix[SINDY_COEFFS].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(presence_rates)), presence_rates.values)
    plt.yticks(range(len(presence_rates)), [c.replace('_', '_\n') for c in presence_rates.index], fontsize=8)
    plt.xlabel('Presence Rate')
    plt.title('Overall SINDy Coefficient Presence Rates')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(presence_rates.values):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'overall_presence_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_coefficient_distributions(df, out_dir):
    """Plot distributions of non-zero SINDy coefficients"""
    
    # Select coefficients with reasonable presence
    presence_rates = df[SINDY_COEFFS].notna().mean()
    coeffs_to_plot = presence_rates[presence_rates > 0.1].index.tolist()
    
    n_cols = 3
    n_rows = int(np.ceil(len(coeffs_to_plot) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, coeff in enumerate(coeffs_to_plot):
        if i < len(axes):
            ax = axes[i]
            data = df[coeff].dropna()
            if len(data) > 0:
                ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
                ax.set_title(f'{coeff.replace("_", " ")[:30]}...', fontsize=10)
                ax.set_xlabel('Coefficient Value')
                ax.set_ylabel('Count')
                ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(coeffs_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'coefficient_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_age_vs_coefficient_values(df, out_dir):
    """Scatter plots of age vs coefficient values"""
    
    presence_rates = df[SINDY_COEFFS].notna().mean()
    top_coeffs = presence_rates.nlargest(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, coeff in enumerate(top_coeffs):
        ax = axes[i]
        mask = df[coeff].notna()
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, 'Age'], df.loc[mask, coeff], alpha=0.6)
            ax.set_xlabel('Age')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f'{coeff.replace("_", " ")[:25]}...', fontsize=10)
            ax.grid(alpha=0.3)
            
            # Add correlation info
            corr = df['Age'].corr(df[coeff])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'age_vs_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_coefficient_count_by_participant(df, out_dir):
    """Plot number of non-zero coefficients per participant"""
    
    coeff_counts = df[SINDY_COEFFS].notna().sum(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of coefficient counts
    ax1.hist(coeff_counts, bins=range(0, coeff_counts.max()+2), alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Non-Zero Coefficients')
    ax1.set_ylabel('Number of Participants')
    ax1.set_title('Distribution of Coefficient Counts per Participant')
    ax1.grid(alpha=0.3)
    
    # Coefficient count vs age
    ax2.scatter(df['Age'], coeff_counts, alpha=0.6)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Number of Non-Zero Coefficients')
    ax2.set_title('Coefficient Count vs Age')
    ax2.grid(alpha=0.3)
    
    # Add correlation
    corr = df['Age'].corr(coeff_counts)
    ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'coefficient_counts.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    df = load_and_clean_data('AAAAsindy_analysis_with_metrics.csv')
    
    # Create output directory
    out_dir = Path('sindy_basic_plots')
    out_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_coefficient_presence_by_age(df, out_dir)
    plot_coefficient_distributions(df, out_dir)
    plot_age_vs_coefficient_values(df, out_dir)
    plot_coefficient_count_by_participant(df, out_dir)
    
    # Print basic stats
    print("Basic Statistics:")
    print(f"Total participants: {len(df)}")
    print(f"Age range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
    print("\nCoefficient presence rates:")
    presence_rates = df[SINDY_COEFFS].notna().mean().sort_values(ascending=False)
    for coeff, rate in presence_rates.head(10).items():
        print(f"  {coeff}: {rate:.3f}")

if __name__ == '__main__':
    main()