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

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Change to the script directory
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# Set pandas option
pd.set_option('future.no_silent_downcasting', True)

# Load the participant features data
features_path = '../features/participant_features.csv'
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
    os.makedirs('data/visualization_plots', exist_ok=True)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    # Professional medical research palette - colorblind friendly
    sns.set_palette("viridis", n_colors=3)  # or try "Set2", "husl", or custom medical colors
    
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
        
        # Boxplot with individual points
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='diagnosis', y=feature, order=diagnosis_order)
        sns.stripplot(data=df, x='diagnosis', y=feature, order=diagnosis_order, 
                     color='black', alpha=0.6, size=4)
        plt.title(f'{feature_labels[feature]} - Box Plot with Individual Points', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Diagnosis', fontsize=14)
        plt.ylabel(feature_labels[feature], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'data/visualization_plots/{plot_counter:02d}_{feature}_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        plot_counter += 1
