import sys
import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

plt.style.use('default')


def create_output_directory(output_dir_path):
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data(data_path):
    """Load the analysis dataframe."""
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} participants")
    print(f"Age range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
    print(f"Age categories: {df['Age_Category'].value_counts().to_dict()}")
    return df


def plot_age_continuous_vs_betas(df, output_dir):
    """Plot continuous age vs beta values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Beta reward vs age
    axes[0].scatter(df['Age'], df['beta_reward'], alpha=0.6, s=50)
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Beta Reward')
    axes[0].set_title('Age vs Beta Reward')
    axes[0].grid(True, alpha=0.3)
    
    z = np.polyfit(df['Age'], df['beta_reward'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['Age'], p(df['Age']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    corr_reward = np.corrcoef(df['Age'], df['beta_reward'])[0, 1]
    axes[0].text(
        0.05, 0.95, f'r = {corr_reward:.3f}',
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    axes[1].scatter(df['Age'], df['beta_choice'], alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel('Age (years)')
    axes[1].set_ylabel('Beta Choice')
    axes[1].set_title('Age vs Beta Choice')
    axes[1].grid(True, alpha=0.3)
    
    z = np.polyfit(df['Age'], df['beta_choice'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['Age'], p(df['Age']), "r--", alpha=0.8, linewidth=2)
    
    corr_choice = np.corrcoef(df['Age'], df['beta_choice'])[0, 1]
    axes[1].text(
        0.05, 0.95, f'r = {corr_choice:.3f}',
        transform=axes[1].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'age_continuous_vs_betas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation Age vs Beta Reward: {corr_reward:.3f}")
    print(f"Correlation Age vs Beta Choice: {corr_choice:.3f}")


def plot_age_category_vs_betas(df, output_dir):
    if df['Age_Category'].isna().all():
        print("Warning: No Age_Category data available")
        return
    
    df_clean = df.dropna(subset=['Age_Category'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Beta reward by age category
    categories = sorted(df_clean['Age_Category'].unique())
    
    # Box plot for beta reward
    box_data_reward = [
        df_clean[df_clean['Age_Category'] == cat]['beta_reward'].values
        for cat in categories
    ]
    bp1 = axes[0].boxplot(box_data_reward, labels=categories, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for i, cat in enumerate(categories):
        cat_data = df_clean[df_clean['Age_Category'] == cat]['beta_reward']
        x = np.random.normal(i + 1, 0.04, size=len(cat_data))
        axes[0].scatter(x, cat_data, alpha=0.6, s=30, color='black')
    
    axes[0].set_xlabel('Age Category')
    axes[0].set_ylabel('Beta Reward')
    axes[0].set_title('Age Category vs Beta Reward')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot for beta choice
    box_data_choice = [
        df_clean[df_clean['Age_Category'] == cat]['beta_choice'].values
        for cat in categories
    ]
    bp2 = axes[1].boxplot(box_data_choice, labels=categories, patch_artist=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for i, cat in enumerate(categories):
        cat_data = df_clean[df_clean['Age_Category'] == cat]['beta_choice']
        x = np.random.normal(i + 1, 0.04, size=len(cat_data))
        axes[1].scatter(x, cat_data, alpha=0.6, s=30, color='black')
    
    axes[1].set_xlabel('Age Category')
    axes[1].set_ylabel('Beta Choice')
    axes[1].set_title('Age Category vs Beta Choice')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'age_category_vs_betas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nBeta Reward by Age Category:")
    print(df_clean.groupby('Age_Category')['beta_reward'].agg(['count', 'mean', 'std']))
    print("\nBeta Choice by Age Category:")
    print(df_clean.groupby('Age_Category')['beta_choice'].agg(['count', 'mean', 'std']))


def main():
    parser = argparse.ArgumentParser(
        description="Generate age vs beta plots from analysis CSV."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/AAAAsindy_analysis_with_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/age_vs_beta_plots",
    )
    args = parser.parse_args()
    
    output_dir = create_output_directory(args.output_dir)
    
    df = load_data(args.data_path)
    
    df_clean = df.dropna(subset=['Age', 'beta_reward', 'beta_choice'])
    
    plot_age_continuous_vs_betas(df_clean, output_dir)
    
    plot_age_category_vs_betas(df, output_dir)


if __name__ == "__main__":
    main()
