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
from scipy.stats import pearsonr

def calculate_correlation(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 2:
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        return f"r = {corr:.3f}"
    return "N/A"

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

def plot_behavioral_vs_metrics(df, output_dir):
    behavioral_measures = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']
    output_metrics     = ['bic_spice',    'nll_spice',       'nll_rnn']
    metric_names       = {
        'bic_spice': 'BIC (SPICE)',
        'nll_spice': 'NLL (SPICE)',
        'nll_rnn':   'NLL (RNN)'
    }

    # Plot 1: Behavioral vs Metrics (by Continuous Age)
    fig = plt.figure(figsize=(15, 12))
    outer_grid = gridspec.GridSpec(
        nrows=len(output_metrics),
        ncols=1,
        height_ratios=[1] * len(output_metrics),
        hspace=0.35
    )

    cmap = plt.cm.viridis
    norm = Normalize(vmin=df['Age'].min(), vmax=df['Age'].max())

    fig.suptitle(
        "Behavioral Measures vs Model Metrics (by Continuous Age)",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Age (years)', fontsize=12)

    for i, metric in enumerate(output_metrics):
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            nrows=1,
            ncols=len(behavioral_measures),
            subplot_spec=outer_grid[i],
            wspace=0.3
        )

        for j, behavior in enumerate(behavioral_measures):
            ax = plt.Subplot(fig, inner_grid[j])
            valid_data = df.dropna(subset=[behavior, metric, 'Age'])

            if len(valid_data) > 2:
                ax.scatter(
                    valid_data[behavior],
                    valid_data[metric],
                    c=valid_data['Age'],
                    cmap=cmap,
                    norm=norm,
                    alpha=0.65,
                    edgecolors='w',
                    linewidth=0.4,
                    s=30
                )

                corr_text = calculate_correlation_with_significance(
                    valid_data[behavior],
                    valid_data[metric]
                )

                sns.regplot(
                    x=behavior,
                    y=metric,
                    data=valid_data,
                    scatter=False,
                    ci=None,
                    line_kws={'color': 'red', 'linewidth': 1.5},
                    ax=ax
                )

                ax.text(
                    0.05, 0.92, corr_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

            ax.set_title(f"{behavior.replace('_', ' ').title()}", fontsize=12)
            ax.set_xlabel(behavior.replace('_', ' ').title(), fontsize=10)

            # Y-axis label only on the first column of each row
            if j == 0:
                ax.set_ylabel(metric_names[metric], fontsize=10)

            fig.add_subplot(ax)

    plt.savefig(
        os.path.join(output_dir, 'behavioral_vs_metrics_continuous_age_fixed.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Plot 2: Behavioral vs Metrics (by Age Category)
    if 'Age_Category' in df.columns and not df['Age_Category'].isna().all():
        fig = plt.figure(figsize=(15, 12))
        outer_grid = gridspec.GridSpec(
            nrows=len(output_metrics),
            ncols=1,
            height_ratios=[1] * len(output_metrics),
            hspace=0.4
        )

        category_colors = {
            1: '#C6E2FF',
            2: '#1E90FF',
            3: '#003366'
        }
        alpha_val   = 0.5
        marker_size = 25

        fig.suptitle(
            "Behavioral Measures vs Model Metrics (by Age Category)",
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        for i, metric in enumerate(output_metrics):
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                nrows=1,
                ncols=len(behavioral_measures),
                subplot_spec=outer_grid[i],
                wspace=0.3
            )

            for j, behavior in enumerate(behavioral_measures):
                ax = plt.Subplot(fig, inner_grid[j])
                valid_data = df.dropna(subset=[behavior, metric, 'Age_Category'])

                if len(valid_data) > 2:
                    for cat, color_hex in category_colors.items():
                        cat_data = valid_data[valid_data['Age_Category'] == cat]
                        if len(cat_data) > 0:
                            ax.scatter(
                                cat_data[behavior],
                                cat_data[metric],
                                c=color_hex,
                                label=f"Age Cat {cat}",
                                alpha=alpha_val,
                                s=marker_size,
                                edgecolors='w',
                                linewidth=0.4
                            )

                    corr_text = calculate_correlation_with_significance(
                        valid_data[behavior],
                        valid_data[metric]
                    )
                    ax.text(
                        0.05, 0.92, corr_text,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                    )

                    # regression line (ignores categories; uses all points)
                    sns.regplot(
                        x=behavior,
                        y=metric,
                        data=valid_data,
                        scatter=False,
                        ci=None,
                        line_kws={'color': 'red', 'linewidth': 1.5},
                        ax=ax
                    )

                    if (i == 0) and (j == len(behavioral_measures) - 1):
                        ax.legend(
                            bbox_to_anchor=(1.05, 1),
                            loc='upper left',
                            title="Age Category",
                            frameon=False
                        )

                ax.set_title(f"{behavior.replace('_', ' ').title()}", fontsize=12)
                ax.set_xlabel(behavior.replace('_', ' ').title(), fontsize=10)

                if j == 0:
                    ax.set_ylabel(metric_names[metric], fontsize=10)

                fig.add_subplot(ax)

        plt.savefig(
            os.path.join(output_dir, 'behavioral_vs_metrics_categorical_age_fixed.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # Plot 3: Correlation Matrix
    plt.figure(figsize=(10, 8))

    selected_columns = behavioral_measures + output_metrics + ['Age']
    df_for_corr = df.copy()

    if 'Age_Category' in df.columns and not df['Age_Category'].isna().all():
        df_for_corr['Age_Category_numeric'] = pd.Categorical(df['Age_Category']).codes
        selected_columns.append('Age_Category_numeric')

    corr_df     = df_for_corr[selected_columns].dropna()
    corr_matrix = corr_df.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        fmt='.2f',
        linewidths=0.5
    )
    plt.title(
        'Correlation Matrix: Behavioral Measures, Model Metrics, and Age',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'correlation_matrix.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print("Statistically Significant Correlations (p < 0.05):")
    print("=" * 50)

    significant_found = False
    for behavior in behavioral_measures:
        for metric in output_metrics:
            valid_data = df.dropna(subset=[behavior, metric])
            if len(valid_data) > 2:
                corr, p_value = pearsonr(valid_data[behavior], valid_data[metric])
                if p_value < 0.05:
                    significant_found = True
                    if p_value < 0.001:
                        sig_level = "p < 0.001"
                    elif p_value < 0.01:
                        sig_level = "p < 0.01"
                    else:
                        sig_level = f"p = {p_value:.3f}"
                    print(f"{behavior} vs {metric}: r = {corr:.3f}, {sig_level}")

    if not significant_found:
        print("No statistically significant correlations found (p < 0.05)")

    print("\nLegend: *** p < 0.001, ** p < 0.01, * p < 0.05")

def main():
    parser = argparse.ArgumentParser(
        description="Generate behavioral performance and age-related plots."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/AAAAsindy_analysis_with_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/behavioral_performance_and_age",
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    plot_behavioral_vs_metrics(df, output_dir)

if __name__ == "__main__":
    main()
