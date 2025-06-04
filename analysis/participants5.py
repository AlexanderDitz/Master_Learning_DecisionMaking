"""
For each SINDy coefficient column, plots its value vs. four behavioral metrics
(switch_rate, stay_after_reward, perseveration, avg_reward) in a 2×2 grid.
Points are color‐coded by participant age.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def main():
    file_path = Path('AAAAsindy_analysis_with_metrics.csv')

    df = pd.read_csv(file_path)
    
    df = df.dropna(subset=['Age'])
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])  
    
    print(f"Age range: {df['Age'].min():.1f} to {df['Age'].max():.1f}")
    print(f"Number of participants: {len(df)}")
    
    exclude_prefixes = [
        'participant_id', 'Age',
        'switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward',
        'beta_reward', 'beta_choice', 'params_', 'total_params',
        'nll_', 'trial_likelihood_', 'bic_', 'aic_',
        'n_parameters_', 'metric_n_trials', 'embedding_', 'n_trials'
    ]
    
    coeffs = [c for c in df.columns
              if not any(c.startswith(pref) for pref in exclude_prefixes)]
    
    behavioral = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']
    
    ages = df['Age'].values
    cmap = 'viridis'
    
    age_min = max(ages.min(),7.8)  
    age_max = min(ages.max(), 30.0)  
    
    output_dir = Path('/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for coeff in coeffs:
        vals = df[coeff].values
        if np.all(np.isnan(vals)) or np.var(vals) == 0:
            print(f"Skipping {coeff} - all NaN or constant values")
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        scatter = None
        
        for ax, metric in zip(axes, behavioral):
            # Skip if metric has issues
            metric_vals = df[metric].values
            if np.all(np.isnan(metric_vals)):
                print(f"Warning: {metric} has all NaN values")
                continue
                
            scatter = ax.scatter(
                vals,
                metric_vals,
                c=ages,
                cmap=cmap,
                vmin=age_min,
                vmax=age_max,
                alpha=0.7,
                s=50,  
                edgecolors='white',
                linewidth=0.5
            )
            
            ax.set_xlabel(coeff, fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f"{metric.replace('_', ' ').title()} vs {coeff}", fontsize=13)
            ax.grid(True, alpha=0.3)
        
        plt.subplots_adjust(
            left=0.08,
            right=0.82, 
            top=0.90,
            bottom=0.10,
            wspace=0.35,
            hspace=0.35
        )
        
        if scatter is not None:
            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Age (years)', rotation=270, labelpad=20, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        
        fig.suptitle(f"{coeff} Coefficient vs Behavioral Metrics", fontsize=16, y=0.95)
        
        safe_coeff = coeff.replace('/', '_').replace('\\', '_').replace(':', '_')
        out_path = output_dir / f"{safe_coeff}_vs_behavior.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()