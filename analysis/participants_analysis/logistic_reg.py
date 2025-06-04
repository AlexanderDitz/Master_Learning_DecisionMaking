import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path='fixed_sindy_analysis_with_metrics.csv'):
    """
    Load data and extract SINDY coefficients for logistic regression analysis
    """
    df = pd.read_csv(csv_path)
    
    # Extract all SINDY coefficient columns
    sindy_cols = [col for col in df.columns if col.startswith('x_')]
    
    print(f"Found {len(sindy_cols)} SINDY coefficient columns")
    
    return df, sindy_cols

def perform_logistic_regression_analysis(df, sindy_cols, output_dir):
    """
    Perform logistic regression for each SINDY coefficient as a function of age
    """
    
    # Extract age data and remove missing values
    valid_age_mask = df['Age'].notna()
    df_clean = df[valid_age_mask].copy()
    age_data = df_clean['Age'].values
    
    print(f"Using {len(df_clean)} participants with complete age data")
    
    # Standardize age for better regression performance
    scaler = StandardScaler()
    age_standardized = scaler.fit_transform(age_data.reshape(-1, 1)).flatten()
    
    # Store results for each coefficient
    results = []
    
    for col in sindy_cols:
        # Create binary outcome: is coefficient non-zero?
        sindy_values = df_clean[col].fillna(0)  # Fill NaN with 0
        is_nonzero = (sindy_values != 0).astype(int)
        
        # Skip if coefficient is always zero or always non-zero
        if is_nonzero.sum() == 0 or is_nonzero.sum() == len(is_nonzero):
            print(f"Skipping {col}: no variation in presence")
            continue
        
        # Perform logistic regression
        try:
            # Using sklearn LogisticRegression
            log_reg = LogisticRegression(fit_intercept=True, max_iter=1000)
            log_reg.fit(age_standardized.reshape(-1, 1), is_nonzero)
            
            # Extract coefficient and intercept
            beta_age = log_reg.coef_[0][0]
            intercept = log_reg.intercept_[0]
            
            # Calculate pseudo R-squared (McFadden's R-squared)
            # Log-likelihood of fitted model
            y_pred_proba = log_reg.predict_proba(age_standardized.reshape(-1, 1))[:, 1]
            log_likelihood = np.sum(is_nonzero * np.log(y_pred_proba + 1e-15) + 
                                  (1 - is_nonzero) * np.log(1 - y_pred_proba + 1e-15))
            
            # Log-likelihood of null model (intercept only)
            p_null = is_nonzero.mean()
            log_likelihood_null = np.sum(is_nonzero * np.log(p_null + 1e-15) + 
                                        (1 - is_nonzero) * np.log(1 - p_null + 1e-15))
            
            pseudo_r2 = 1 - (log_likelihood / log_likelihood_null)
            
            # Calculate p-value using likelihood ratio test approximation
            # For large samples, -2 * log(likelihood ratio) ~ chi-squared with 1 df
            lr_stat = -2 * (log_likelihood_null - log_likelihood)
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lr_stat, df=1)
            
            # Calculate basic statistics
            presence_rate = is_nonzero.mean()
            
            # Store results
            results.append({
                'coefficient': col,
                'beta_age': beta_age,
                'intercept': intercept,
                'p_value': p_value,
                'pseudo_r2': pseudo_r2,
                'presence_rate': presence_rate,
                'n_nonzero': is_nonzero.sum(),
                'n_total': len(is_nonzero),
                'coefficient_clean': clean_coefficient_name(col)
            })
            
        except Exception as e:
            print(f"Error processing {col}: {e}")
            continue
    
    # Convert to DataFrame and sort by beta coefficient
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("No valid coefficients for analysis!")
        return
    
    results_df = results_df.sort_values('beta_age', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'sindy_age_logistic_regression_results.csv'), index=False)
    
    # Create visualization
    create_logistic_regression_plot(results_df, output_dir)
    
    return results_df

def clean_coefficient_name(col_name):
    """
    Clean up coefficient names for better display
    """
    # Remove the 'x_' prefix
    clean_name = col_name.replace('x_', '')
    
    # Replace underscores with spaces and title case
    clean_name = clean_name.replace('_', ' ').title()
    
    # Special handling for common patterns
    clean_name = clean_name.replace('C ', 'Control ')
    clean_name = clean_name.replace(' 1', ' (Bias)')
    
    # Truncate if too long
    if len(clean_name) > 100:
        clean_name = clean_name[:27] + '...'
    
    return clean_name

def create_logistic_regression_plot(results_df, output_dir):
    """
    Create visualization of logistic regression results
    """
    
    # Determine significance levels
    results_df['significance'] = 'ns'
    results_df.loc[results_df['p_value'] < 0.05, 'significance'] = '*'
    results_df.loc[results_df['p_value'] < 0.01, 'significance'] = '**'
    results_df.loc[results_df['p_value'] < 0.001, 'significance'] = '***'
    
    # Color mapping for significance
    color_map = {'ns': '#CCCCCC', '*': '#FFB347', '**': '#FF6B47', '***': '#FF1744'}
    colors = [color_map[sig] for sig in results_df['significance']]
    
    # Create the main plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Age-Dependent Presence of SINDY Coefficients: Logistic Regression Analysis', 
                 fontsize=16, y=0.98)
    
    # 1. Main beta coefficient plot
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results_df))
    bars = ax1.bar(x_pos, results_df['beta_age'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars, results_df['significance'])):
        if sig != 'ns':
            height = bar.get_height()
            y_pos = height + 0.01 if height > 0 else height - 0.02
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos, sig, 
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=10, fontweight='bold')
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('SINDY Coefficients (ordered by effect size)')
    ax1.set_ylabel('Beta Coefficient (Age Effect)')
    ax1.set_title('Age Effect on Coefficient Presence')
    ax1.set_xticks(x_pos[::max(1, len(x_pos)//10)])  # Show every nth label
    ax1.set_xticklabels([results_df.iloc[i]['coefficient_clean'] for i in x_pos[::max(1, len(x_pos)//10)]], 
                       rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. P-value distribution
    ax2 = axes[0, 1]
    ax2.hist(results_df['p_value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(0.05, color='red', linestyle='--', label='p = 0.05')
    ax2.axvline(0.01, color='orange', linestyle='--', label='p = 0.01')
    ax2.axvline(0.001, color='darkred', linestyle='--', label='p = 0.001')
    ax2.set_xlabel('P-value')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of P-values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Beta vs presence rate
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['presence_rate'], results_df['beta_age'], 
                         c=[color_map[sig] for sig in results_df['significance']], 
                         alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Coefficient Presence Rate')
    ax3.set_ylabel('Beta Coefficient (Age Effect)')
    ax3.set_title('Age Effect vs Coefficient Prevalence')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Significance summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary statistics
    sig_counts = results_df['significance'].value_counts()
    total_coeffs = len(results_df)
    
    summary_text = f"""Logistic Regression Summary
    
Total Coefficients Analyzed: {total_coeffs}

Significance Levels:
• p < 0.001 (***): {sig_counts.get('***', 0)} ({sig_counts.get('***', 0)/total_coeffs*100:.1f}%)
• p < 0.01 (**): {sig_counts.get('**', 0)} ({sig_counts.get('**', 0)/total_coeffs*100:.1f}%)
• p < 0.05 (*): {sig_counts.get('*', 0)} ({sig_counts.get('*', 0)/total_coeffs*100:.1f}%)
• Non-significant: {sig_counts.get('ns', 0)} ({sig_counts.get('ns', 0)/total_coeffs*100:.1f}%)

Age Range: {results_df.iloc[0] if len(results_df) > 0 else 'N/A'}

Beta Range: {results_df['beta_age'].min():.3f} to {results_df['beta_age'].max():.3f}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add legend for significance colors
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', 
                                   label=f'{sig} (p {threshold})') 
                      for sig, threshold, color in [
                          ('***', '< 0.001', '#FF1744'),
                          ('**', '< 0.01', '#FF6B47'), 
                          ('*', '< 0.05', '#FFB347'),
                          ('ns', '≥ 0.05', '#CCCCCC')
                      ]]
    
    ax4.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'sindy_age_logistic_regression_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Logistic regression plot saved to: {os.path.join(output_dir, 'sindy_age_logistic_regression_analysis.png')}")

def create_detailed_coefficient_plot(results_df, output_dir):
    """
    Create a detailed plot showing individual coefficient results
    """
    
    # Create a horizontal bar plot for better readability
    fig, ax = plt.subplots(figsize=(12, max(8, len(results_df) * 0.4)))
    
    # Color by significance
    color_map = {'ns': '#CCCCCC', '*': '#FFB347', '**': '#FF6B47', '***': '#FF1744'}
    colors = [color_map[sig] for sig in results_df['significance']]
    
    y_pos = np.arange(len(results_df))
    bars = ax.barh(y_pos, results_df['beta_age'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add coefficient names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['coefficient_clean'], fontsize=10)
    ax.set_xlabel('Beta Coefficient (Age Effect on Presence)')
    ax.set_title('Age Effects on Individual SINDY Coefficient Presence\n(Ordered by Effect Size)', 
                fontsize=14, pad=20)
    
    # Add significance markers
    for i, (bar, sig, p_val) in enumerate(zip(bars, results_df['significance'], results_df['p_value'])):
        if sig != 'ns':
            width = bar.get_width()
            x_pos = width + 0.01 if width > 0 else width - 0.02
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{sig}\n(p={p_val:.3f})', 
                   ha='left' if width > 0 else 'right', va='center', 
                   fontsize=8, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', 
                                   label=f'{sig}') 
                      for sig, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save the detailed plot
    plt.savefig(os.path.join(output_dir, 'sindy_age_detailed_coefficient_effects.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed coefficient plot saved to: {os.path.join(output_dir, 'sindy_age_detailed_coefficient_effects.png')}")

def print_summary_results(results_df):
    """
    Print summary of logistic regression results
    """
    
    print("\n" + "="*80)
    print("SINDY COEFFICIENT AGE-DEPENDENCY ANALYSIS SUMMARY")
    print("="*80)
    
    sig_results = results_df[results_df['p_value'] < 0.05].sort_values('p_value')
    
    if len(sig_results) > 0:
        print(f"\nSignificant Age Effects Found in {len(sig_results)} coefficients:")
        print("-" * 60)
        for _, row in sig_results.iterrows():
            direction = "increases" if row['beta_age'] > 0 else "decreases"
            print(f"• {row['coefficient_clean']}")
            print(f"  β = {row['beta_age']:.3f}, p = {row['p_value']:.3f} {row['significance']}")
            print(f"  Presence {direction} with age (prevalence: {row['presence_rate']:.1%})")
            print()
    else:
        print("\nNo significant age effects found in any SINDY coefficients.")
    
    print(f"\nOverall Statistics:")
    print(f"• Total coefficients analyzed: {len(results_df)}")
    print(f"• Significant at p < 0.05: {(results_df['p_value'] < 0.05).sum()}")
    print(f"• Significant at p < 0.01: {(results_df['p_value'] < 0.01).sum()}")
    print(f"• Significant at p < 0.001: {(results_df['p_value'] < 0.001).sum()}")
    print(f"• Beta coefficient range: {results_df['beta_age'].min():.3f} to {results_df['beta_age'].max():.3f}")

def main():
    # Set up output directory
    output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy_age_logistic'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("Loading data...")
    df, sindy_cols = load_and_prepare_data()
    
    print(f"Dataset shape: {df.shape}")
    
    # Perform logistic regression analysis
    print("\nPerforming logistic regression analysis...")
    results_df = perform_logistic_regression_analysis(df, sindy_cols, output_dir)
    
    if results_df is not None and len(results_df) > 0:
        # Create detailed coefficient plot
        print("\nCreating detailed coefficient plot...")
        create_detailed_coefficient_plot(results_df, output_dir)
        
        # Print summary
        print_summary_results(results_df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()