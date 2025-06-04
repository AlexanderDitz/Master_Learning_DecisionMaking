import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, f_oneway
import warnings

warnings.filterwarnings('ignore')

output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots/sindy_coefficients_analysis'
os.makedirs(output_dir, exist_ok=True)

# Set consistent style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def load_and_prepare_data(csv_path='AAAAsindy_analysis_with_metrics.csv'):
    df = pd.read_csv(csv_path)

    # RNN modules
    list_rnn_modules = [
        'x_learning_rate_reward',
        'x_value_reward_not_chosen',
        'x_value_choice_chosen',
        'x_value_choice_not_chosen'
    ]

    all_sindy_cols = [col for col in df.columns if col.startswith('x_')]

    modules = {}
    for module in list_rnn_modules:
        modules[module] = {
            'bias': [],
            'variable': [],
            'control': [],
            'variable_interaction': []
        }

    # Categorize SINDY columns by module and input type
    for col in all_sindy_cols:
        module_found = None
        for module in list_rnn_modules:
            if col.startswith(module):
                module_found = module
                break
        if module_found is None:
            continue

        if col.endswith('_1'):
            modules[module_found]['bias'].append(col)
        elif '_x_' in col and not col.endswith('_1'):
            modules[module_found]['variable'].append(col)
        elif 'c_action' in col or 'c_reward' in col:
            modules[module_found]['control'].append(col)
        elif 'c_value_reward' in col or 'c_value_choice' in col:
            modules[module_found]['variable_interaction'].append(col)
        else:
            if '_x_' in col:
                modules[module_found]['variable'].append(col)

    modules = {
        k: v for k, v in modules.items()
        if any(len(input_list) > 0 for input_list in v.values())
    }

    beta_columns = [c for c in ['beta_reward', 'beta_choice'] if c in df.columns]
    param_count_columns = [c for c in df.columns if c.startswith('params_')]

    behavioral_columns = [
        c for c in [
            'stay_after_reward',
            'switch_rate',
            'perseveration',
            'stay_after_plus_plus',
            'stay_after_plus_minus',
            'stay_after_minus_plus',
            'stay_after_minus_minus',
            'avg_reward',
            'avg_rt'
        ] if c in df.columns
    ]

    all_sindy_flat = []
    for module_data in modules.values():
        for input_type_list in module_data.values():
            all_sindy_flat.extend(input_type_list)

    return df, {
        'modules': modules,
        'beta': beta_columns,
        'param_counts': param_count_columns,
        'behavioral': behavioral_columns,
        'all_sindy': all_sindy_flat
    }


def sindy_module_analysis(df, columns_dict, output_dir):
    """
    Analyze SINDY coefficients by module and input type.
    Creates visualizations showing:
    1. Activity patterns across modules
    2. Input type distributions within modules
    3. Coefficient magnitude distributions
    All bar plots include SEM error bars.
    """

    modules = columns_dict['modules']
    if not modules:
        print("No SINDY modules found.")
        return

    n_participants = df.shape[0]
    module_stats = {}

    # Compute per-participant metrics to derive SEMs
    for module_name, input_types in modules.items():
        module_stats[module_name] = {}
        for input_type, cols in input_types.items():
            if len(cols) == 0:
                continue

            # Activity per participant
            activity_per_part = (df[cols] != 0).sum(axis=1) / float(len(cols))
            mean_activity = activity_per_part.mean()
            sem_activity = activity_per_part.std(ddof=1) / np.sqrt(n_participants)

            # Mean absolute coefficient per participant (ignoring zeros)
            abs_vals = df[cols].abs()
            mean_abs_per_part = []
            for i in range(n_participants):
                nonzero_vals = abs_vals.iloc[i][abs_vals.iloc[i] != 0].values
                if len(nonzero_vals) > 0:
                    mean_abs_per_part.append(nonzero_vals.mean())
                else:
                    mean_abs_per_part.append(np.nan)
            mean_abs_per_part = np.array(mean_abs_per_part)
            valid_mask = ~np.isnan(mean_abs_per_part)
            if valid_mask.sum() > 0:
                mean_abs_coef = np.nanmean(mean_abs_per_part)
                sem_abs_coef = mean_abs_per_part[valid_mask].std(ddof=1) / np.sqrt(valid_mask.sum())
            else:
                mean_abs_coef = 0.0
                sem_abs_coef = 0.0

            # Number of features is fixed across participants; SEM = 0
            n_features = len(cols)
            sem_n_features = 0.0

            module_stats[module_name][input_type] = {
                'n_features': n_features,
                'sem_n_features': sem_n_features,
                'mean_activity': mean_activity,
                'sem_activity': sem_activity,
                'mean_abs_coef': mean_abs_coef,
                'sem_abs_coef': sem_abs_coef
            }

    # Create visualization with SEM error bars
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SINDY Module Analysis', fontsize=16)

    # 1. Activity rates by module and input type
    ax1 = axes[0, 0]
    module_names = list(modules.keys())
    input_types = ['bias', 'variable', 'control', 'variable_interaction']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    x = np.arange(len(module_names))
    width = 0.2

    for i, input_type in enumerate(input_types):
        activities = []
        errs = []
        for module in module_names:
            stats = module_stats[module].get(input_type, None)
            if stats is not None:
                activities.append(stats['mean_activity'])
                errs.append(stats['sem_activity'])
            else:
                activities.append(0.0)
                errs.append(0.0)
        ax1.bar(
            x + i * width,
            activities,
            width,
            yerr=errs,
            capsize=5,
            label=input_type.replace('_', ' ').title(),
            color=colors[i],
            alpha=0.7
        )

    ax1.set_xlabel('Module')
    ax1.set_ylabel('Activity Rate')
    ax1.set_title('Activity Rate by Module and Input Type')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(
        [name.replace('x_', '').replace('_', ' ') for name in module_names],
        rotation=45,
        ha='right'
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Number of features by module and input type (SEM = 0)
    ax2 = axes[0, 1]
    for i, input_type in enumerate(input_types):
        n_features = []
        errs = []
        for module in module_names:
            stats = module_stats[module].get(input_type, None)
            if stats is not None:
                n_features.append(stats['n_features'])
                errs.append(stats['sem_n_features'])
            else:
                n_features.append(0)
                errs.append(0.0)
        ax2.bar(
            x + i * width,
            n_features,
            width,
            yerr=errs,
            capsize=5,
            label=input_type.replace('_', ' ').title(),
            color=colors[i],
            alpha=0.7
        )

    ax2.set_xlabel('Module')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Count by Module and Input Type')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(
        [name.replace('x_', '').replace('_', ' ') for name in module_names],
        rotation=45,
        ha='right'
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Mean coefficient magnitudes with SEM
    ax3 = axes[1, 0]
    for i, input_type in enumerate(input_types):
        mean_coefs = []
        errs = []
        for module in module_names:
            stats = module_stats[module].get(input_type, None)
            if stats is not None:
                mean_coefs.append(stats['mean_abs_coef'])
                errs.append(stats['sem_abs_coef'])
            else:
                mean_coefs.append(0.0)
                errs.append(0.0)
        ax3.bar(
            x + i * width,
            mean_coefs,
            width,
            yerr=errs,
            capsize=5,
            label=input_type.replace('_', ' ').title(),
            color=colors[i],
            alpha=0.7
        )

    ax3.set_xlabel('Module')
    ax3.set_ylabel('Mean |Coefficient|')
    ax3.set_title('Mean Coefficient Magnitude by Module and Input Type')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(
        [name.replace('x_', '').replace('_', ' ') for name in module_names],
        rotation=45,
        ha='right'
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Overall module activity (mean ± SEM across participants)
    ax4 = axes[1, 1]
    overall_means = []
    overall_sems = []
    for module_name in module_names:
        all_cols = []
        for cols in modules[module_name].values():
            all_cols.extend(cols)
        if len(all_cols) > 0:
            activity_per_part = (df[all_cols] != 0).sum(axis=1) / float(len(all_cols))
            overall_means.append(activity_per_part.mean())
            overall_sems.append(activity_per_part.std(ddof=1) / np.sqrt(n_participants))
        else:
            overall_means.append(0.0)
            overall_sems.append(0.0)

    bars = ax4.bar(
        range(len(module_names)),
        overall_means,
        yerr=overall_sems,
        capsize=5,
        color='purple',
        alpha=0.7
    )
    ax4.set_xlabel('Module')
    ax4.set_ylabel('Overall Activity Rate')
    ax4.set_title('Overall Activity Rate by Module')
    ax4.set_xticks(range(len(module_names)))
    ax4.set_xticklabels(
        [name.replace('x_', '').replace('_', ' ') for name in module_names],
        rotation=45,
        ha='right'
    )
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, overall_means):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sindy_module_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\n====== SINDY MODULE ANALYSIS SUMMARY ======")
    for module_name, input_types in modules.items():
        print(f"\n{module_name.replace('x_', '').replace('_', ' ').title()}:")
        for input_type, cols in input_types.items():
            if len(cols) == 0:
                continue
            stats = module_stats[module_name][input_type]
            print(f"  {input_type.replace('_', ' ').title()}:")
            print(f"    Features: {stats['n_features']}")
            print(f"    Activity rate: {stats['mean_activity']:.3f} ± {stats['sem_activity']:.3f}")
            print(f"    Mean |coef|: {stats['mean_abs_coef']:.3f} ± {stats['sem_abs_coef']:.3f}")


def sindy_behavioral_analysis(df, columns_dict, output_dir):
    """
    Updated behavioral analysis using the modular structure.
    Analyzes correlations between modules/input types and behavioral measures.
    Now includes SEM error bars for each correlation in horizontal bar plots.
    """
    behavioral_cols = columns_dict['behavioral']
    modules = columns_dict['modules']
    beta_cols = columns_dict['beta']

    # Create groups for analysis - including modular breakdown
    analysis_groups = {}

    # Add beta parameters as a group
    if len(beta_cols) > 0:
        analysis_groups['Beta Parameters'] = beta_cols

    # Add each module's input types as separate groups
    for module_name, input_types in modules.items():
        clean_module_name = module_name.replace('x_', '').replace('_', ' ').title()
        for input_type, cols in input_types.items():
            if len(cols) > 0:
                group_name = f"{clean_module_name} - {input_type.replace('_', ' ').title()}"
                analysis_groups[group_name] = cols

    # Compute correlations and SEM for each (feature, behavior) pair
    all_correlations = []
    for group_name, feature_cols in analysis_groups.items():
        for feature_col in feature_cols:
            if feature_col not in df.columns:
                continue
            for behav_col in behavioral_cols:
                if behav_col not in df.columns:
                    continue
                pair = df[[feature_col, behav_col]].dropna()
                n = len(pair)
                if n > 20:
                    r, p = pearsonr(pair[feature_col], pair[behav_col])
                    r_abs = abs(r)
                    # Standard error of Pearson r: sqrt((1 - r^2) / (n - 2))
                    sem_r = np.sqrt((1 - r_abs**2) / (n - 2))
                    all_correlations.append({
                        'Feature_Group': group_name,
                        'Feature': feature_col,
                        'Behavioral_Measure': behav_col,
                        'Correlation': r,
                        'Abs_Correlation': r_abs,
                        'P_Value': p,
                        'N_Samples': n,
                        'SEM': sem_r
                    })

    corr_df = pd.DataFrame(all_correlations)
    if corr_df.empty:
        print("No valid (feature, behavior) pairs to correlate.")
        return corr_df

    # Sort by absolute correlation
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df.to_csv(os.path.join(output_dir, 'sindy_modular_behavioral_correlations.csv'), index=False)

    # Create summary visualization of strongest correlations by module group
    sindy_groups = {k: v for k, v in analysis_groups.items() if 'Beta' not in k}
    if len(sindy_groups) == 0:
        return corr_df

    n_groups = len(sindy_groups)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    fig.suptitle('Strongest SINDY Module-Behavioral Correlations', fontsize=16)

    # Flatten axes array for easy indexing
    if n_groups == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, (group_name, _) in enumerate(sindy_groups.items()):
        ax = axes[i]
        subset = corr_df[corr_df['Feature_Group'] == group_name]
        if subset.empty:
            ax.text(0.5, 0.5, f'No {group_name}\nData', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        top_corrs = subset.head(5)
        y_pos = np.arange(len(top_corrs))
        corrs = top_corrs['Correlation'].values
        sems = top_corrs['SEM'].values
        labels = []
        p_vals = top_corrs['P_Value'].values

        for _, row in top_corrs.iterrows():
            behav_name = row['Behavioral_Measure'].replace('_', ' ').replace('avg rt', 'Average RT')
            labels.append(behav_name)

        bars = ax.barh(y_pos, corrs, xerr=sems, capsize=5, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Correlation')
        ax.set_title(group_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Add significance stars to the right of each bar
        for j, (bar, p_val) in enumerate(zip(bars, p_vals)):
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                marker = ''
            if marker:
                x_pos = bar.get_width() + (0.02 if bar.get_width() > 0 else -0.02)
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, marker,
                        va='center', fontsize=10)

    # Turn off any unused subplots
    for idx in range(len(sindy_groups), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sindy_modular_behavioral_correlations.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return corr_df


def sindy_age_analysis(df, columns_dict, output_dir):
    """
    Updated age analysis using the modular structure.
    """
    modules = columns_dict['modules']
    beta_cols = columns_dict['beta']

    analysis_groups = {}
    if len(beta_cols) > 0:
        analysis_groups['Beta Parameters'] = beta_cols

    for module_name, input_types in modules.items():
        clean_module_name = module_name.replace('x_', '').replace('_', ' ').title()
        all_cols = []
        for cols in input_types.values():
            all_cols.extend(cols)
        if len(all_cols) > 0:
            analysis_groups[clean_module_name] = all_cols

    age_correlations = []
    for group_name, feature_cols in analysis_groups.items():
        for feature_col in feature_cols:
            if feature_col not in df.columns:
                continue
            subset = df[['Age', feature_col]].dropna()
            n = len(subset)
            if n > 20:
                r, p = pearsonr(subset['Age'], subset[feature_col])
                age_correlations.append({
                    'Feature_Group': group_name,
                    'Feature': feature_col,
                    'Correlation': r,
                    'P_Value': p,
                    'N_Samples': n,
                    'Abs_Correlation': abs(r)
                })

    age_corr_df = pd.DataFrame(age_correlations)
    if age_corr_df.empty:
        print("No valid (Age, feature) pairs to correlate.")
        return age_corr_df

    age_corr_df = age_corr_df.sort_values('Abs_Correlation', ascending=False)
    age_corr_df.to_csv(os.path.join(output_dir, 'sindy_modular_age_correlations.csv'), index=False)
    return age_corr_df


def sindy_overview(df, columns_dict, output_dir):
    """
    Updated overview using the modular structure.
    Includes SEM for the sparsity mean line and module activity bars.
    """
    modules = columns_dict['modules']
    all_sindy_cols = columns_dict['all_sindy']
    param_count_cols = columns_dict['param_counts']

    if len(all_sindy_cols) == 0:
        print("No SINDY coefficient columns found – skipping overview.")
        return

    n_participants = df.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SINDY Modular Overview', fontsize=18)

    # 1. Overall sparsity per participant
    sindy_data = df[all_sindy_cols]
    sparsity = (sindy_data != 0).sum(axis=1) / float(len(all_sindy_cols))
    ax0 = axes[0, 0]
    ax0.hist(sparsity, bins=25, color='purple', alpha=0.7, edgecolor='black')
    ax0.set_xlabel('Proportion of Non-zero Coefficients')
    ax0.set_ylabel('Number of Participants')
    ax0.set_title('Overall SINDY Sparsity Distribution')
    ax0.grid(True, alpha=0.3)
    mean_sparsity = np.mean(sparsity)
    sem_sparsity = np.std(sparsity, ddof=1) / np.sqrt(n_participants)
    ax0.axvline(mean_sparsity, color='red', linestyle='--',
                label=f'Mean: {mean_sparsity:.3f} ± {sem_sparsity:.3f}')
    ax0.legend()

    # 2. Activity by module (mean ± SEM)
    ax1 = axes[0, 1]
    module_names = list(modules.keys())
    module_means = []
    module_sems = []

    for module_name in module_names:
        all_cols = []
        for cols in modules[module_name].values():
            all_cols.extend(cols)
        if len(all_cols) > 0:
            activity_per_part = (df[all_cols] != 0).sum(axis=1) / float(len(all_cols))
            module_means.append(activity_per_part.mean())
            module_sems.append(activity_per_part.std(ddof=1) / np.sqrt(n_participants))
        else:
            module_means.append(0.0)
            module_sems.append(0.0)

    bars = ax1.bar(
        range(len(module_names)),
        module_means,
        yerr=module_sems,
        capsize=5,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(module_names)],
        alpha=0.7
    )
    ax1.set_ylabel('Activity Rate')
    ax1.set_title('Activity Rate by Module')
    ax1.set_xticks(range(len(module_names)))
    ax1.set_xticklabels(
        [name.replace('x_', '').replace('_', ' ') for name in module_names],
        rotation=45,
        ha='right'
    )
    ax1.grid(True, alpha=0.3)

    for bar, val in zip(bars, module_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Distribution of all nonzero SINDY coefficients
    ax2 = axes[0, 2]
    all_nonzero = sindy_data.values[sindy_data.values != 0]
    if len(all_nonzero) > 0:
        ax2.hist(all_nonzero, bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Coefficient Value')
        ax2.set_ylabel('Count')
        ax2.set_title('All Non-zero Coefficients Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(all_nonzero), color='red', linestyle='--',
                    label=f'Mean: {np.mean(all_nonzero):.3f}')
        ax2.axvline(np.median(all_nonzero), color='green', linestyle='--',
                    label=f'Median: {np.median(all_nonzero):.3f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Non-zero Coefficients', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_xticks([]); ax2.set_yticks([])

    # 4-6. Feature counts by input type for top 3 modules (pie charts)
    input_types = ['bias', 'variable', 'control', 'variable_interaction']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, module_name in enumerate(module_names[:3]):
        ax = axes[1, i]
        input_counts = []
        input_labels = []
        for input_type in input_types:
            if input_type in modules[module_name]:
                count = len(modules[module_name][input_type])
                if count > 0:
                    input_counts.append(count)
                    input_labels.append(input_type.replace('_', ' ').title())

        if len(input_counts) > 0:
            ax.pie(input_counts, labels=input_labels, autopct='%1.1f%%',
                   colors=colors[:len(input_counts)])
            ax.set_title(f'{module_name.replace("x_", "").replace("_", " ").title()}\nInput Types')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sindy_modular_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def beta_behavior_age_analysis(df, output_dir):
    """
    Enhanced analysis of Beta-Behavior correlations by Age Category
    with SEM error bars.
    """

    # Define better behavioral groupings
    reward_behaviors = [
        'stay_after_plus_plus',   # Stay after reward-reward
        'stay_after_plus_minus',  # Stay after reward-no reward
        'stay_after_reward'       # Overall stay after reward
    ]

    punishment_behaviors = [
        'stay_after_minus_plus',   # Stay after no reward-reward
        'stay_after_minus_minus',  # Stay after no reward-no reward
    ]

    general_behaviors = [
        'switch_rate',
        'perseveration',
        'avg_reward'
    ]

    beta_cols = ['beta_reward', 'beta_choice']

    # Fix age category ordering - ensure proper order
    age_groups = df['Age_Category'].dropna().unique()

    # Sort age groups properly (assuming they're like "Category 1", "Category 2", etc.)
    if all('Category' in str(age) for age in age_groups):
        age_groups = sorted(age_groups, key=lambda x: int(str(x).split()[-1]))
    else:
        age_groups = sorted(age_groups)

    print(f"Age groups in order: {age_groups}")

    # Create comprehensive analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Beta Parameters vs Behavioral Measures by Age Category', fontsize=16)

    behavior_groups = {
        'Reward-Related': reward_behaviors,
        'Punishment-Related': punishment_behaviors,
        'General': general_behaviors
    }

    colors = ['#2E8B57', '#DC143C', '#4169E1', '#FF8C00', '#9932CC']

    for beta_idx, beta in enumerate(beta_cols):
        for group_idx, (group_name, behaviors) in enumerate(behavior_groups.items()):
            ax = axes[beta_idx, group_idx]

            # Filter behaviors that actually exist in the dataframe
            available_behaviors = [b for b in behaviors if b in df.columns]
            if not available_behaviors:
                ax.text(0.5, 0.5, f'No {group_name}\nData Available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x = np.arange(len(age_groups))
            width = 0.8 / len(available_behaviors)

            max_corr = 0.0  # Track max correlation for y-axis scaling
            corr_matrix = []
            sem_matrix = []

            # First pass: compute correlations and SEMs for each (age, behavior)
            for behavior in available_behaviors:
                corr_vals = []
                sem_vals = []
                for age in age_groups:
                    subset = df[df['Age_Category'] == age][[beta, behavior]].dropna()
                    n = len(subset)
                    if n >= 5:
                        r, _ = pearsonr(subset[beta], subset[behavior])
                        r_abs = abs(r)
                        corr_vals.append(r_abs)
                        # SEM of Pearson r: sqrt((1 - r^2) / (n - 2))
                        sem_r = np.sqrt((1 - r_abs**2) / (n - 2))
                        sem_vals.append(sem_r)
                    else:
                        corr_vals.append(0.0)
                        sem_vals.append(0.0)
                corr_matrix.append(corr_vals)
                sem_matrix.append(sem_vals)
                max_corr = max(max_corr, np.max(corr_vals) if corr_vals else 0.0)

            # Second pass: plot bars with error bars
            for i, behavior in enumerate(available_behaviors):
                corr_vals = corr_matrix[i]
                sem_vals = sem_matrix[i]
                bar_positions = x + i * width - width * (len(available_behaviors) - 1) / 2
                ax.bar(
                    bar_positions,
                    corr_vals,
                    width,
                    yerr=sem_vals,
                    capsize=5,
                    label=behavior.replace('_', ' ').replace('stay after', 'Stay:').title(),
                    color=colors[i % len(colors)],
                    alpha=0.7
                )

            ax.set_xlim(-0.5, len(age_groups) - 0.5)
            ax.set_ylim(0, max(max_corr * 1.15, 0.1))
            ax.set_xticks(x)
            ax.set_xticklabels(age_groups, rotation=0)
            ax.set_ylabel(f'|Correlation| with {beta.replace("_", " ").title()}')
            ax.set_title(f'{group_name} Behaviors')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_beta_behavior_age_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create a summary heatmap (unchanged)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Beta-Behavior Correlation Heatmaps by Age Category', fontsize=14)

    for beta_idx, beta in enumerate(beta_cols):
        ax = axes[beta_idx]

        # Collect all available behaviors
        all_behaviors = []
        for behaviors in behavior_groups.values():
            all_behaviors.extend([b for b in behaviors if b in df.columns])

        # Create correlation matrix
        corr_matrix = np.zeros((len(age_groups), len(all_behaviors)))

        for i, age in enumerate(age_groups):
            for j, behavior in enumerate(all_behaviors):
                subset = df[df['Age_Category'] == age][[beta, behavior]].dropna()
                if len(subset) >= 5:
                    r, _ = pearsonr(subset[beta], subset[behavior])
                    corr_matrix[i, j] = abs(r)
                else:
                    corr_matrix[i, j] = np.nan

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(all_behaviors)))
        ax.set_xticklabels([b.replace('_', ' ').title() for b in all_behaviors],
                           rotation=45, ha='right')
        ax.set_yticks(range(len(age_groups)))
        ax.set_yticklabels(age_groups)
        ax.set_title(f'{beta.replace("_", " ").title()}')

        # Add correlation values as text
        for i in range(len(age_groups)):
            for j in range(len(all_behaviors)):
                if not np.isnan(corr_matrix[i, j]):
                    ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha="center", va="center",
                            color="black" if corr_matrix[i, j] < 0.5 else "white")

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('|Correlation|')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'beta_behavior_correlation_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("Enhanced beta-behavior analysis plots saved!")


def scatter_switch_rate_vs_beta(df, output_dir):
    """
    Scatter plots of Beta Reward and Beta Choice vs. switch_rate, avg_rt, perseveration,
    colored by continuous Age.
    - First row: y = beta_reward
    - Second row: y = beta_choice
    - Columns: x = switch_rate, avg_rt, perseveration
    """

    required_cols = ['Age', 'switch_rate', 'avg_rt', 'perseveration', 'beta_reward', 'beta_choice']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing columns for scatter analysis: {missing}. Skipping scatter plot.")
        return

    # Drop rows with NaN in any required column
    scatter_df = df.dropna(subset=required_cols)

    # Setup subplots: 2 rows (beta_reward, beta_choice) x 3 columns (switch_rate, avg_rt, perseveration)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=False)
    fig.suptitle('Scatter: Beta vs Behavioral Measures, Colored by Age', fontsize=16)

    x_vars = ['switch_rate', 'avg_rt', 'perseveration']
    y_vars = ['beta_reward', 'beta_choice']
    y_titles = ['Beta Reward', 'Beta Choice']
    x_titles = ['Switch Rate', 'Average RT', 'Perseveration']

    # Normalize Age for colormap
    ages = scatter_df['Age']
    norm = plt.Normalize(vmin=ages.min(), vmax=ages.max())
    cmap = plt.cm.viridis

    for row_idx, (y_var, y_title) in enumerate(zip(y_vars, y_titles)):
        for col_idx, (x_var, x_title) in enumerate(zip(x_vars, x_titles)):
            ax = axes[row_idx, col_idx]
            sc = ax.scatter(
                scatter_df[x_var],
                scatter_df[y_var],
                c=scatter_df['Age'],
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                edgecolor='k',
                linewidth=0.3
            )
            ax.set_xlabel(x_title)
            ax.set_ylabel(y_title)
            ax.set_title(f'{y_title} vs {x_title}')
            ax.grid(True, alpha=0.3)

    # Add a single colorbar on the right
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        orientation='vertical',
        fraction=0.02,
        pad=0.04
    )
    cbar.set_label('Age')

    # Adjust layout to accommodate title and colorbar
    plt.subplots_adjust(top=0.92, right=0.90, left=0.07, bottom=0.07, wspace=0.25, hspace=0.3)

    plt.savefig(os.path.join(output_dir, 'scatter_beta_vs_behavior_age_colored.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatter plot of Beta vs behavioral measures saved!")


def main():
    print(f"Output directory: {output_dir}")
    df, columns_dict = load_and_prepare_data()

    print(f"\nDataset shape: {df.shape}")
    print("\n" + "=" * 80)
    print("DETAILED SINDY MODULAR STRUCTURE")
    print("=" * 80)

    modules = columns_dict['modules']
    for module_name, input_types in modules.items():
        print(f"\n MODULE: {module_name}")
        print("-" * (len(module_name) + 10))

        total_cols_in_module = 0
        for input_type, cols in input_types.items():
            if len(cols) > 0:
                total_cols_in_module += len(cols)
                print(f"  🔹 {input_type.upper()} ({len(cols)} columns):")
                for col in cols:
                    print(f"    • {col}")

        if total_cols_in_module == 0:
            print("    (No columns found)")
        else:
            print(f"  Total columns in module: {total_cols_in_module}")

    print(f"\n" + "=" * 80)
    print("OTHER COLUMN TYPES")
    print("=" * 80)

    print(f"\n BETA PARAMETERS ({len(columns_dict['beta'])} columns):")
    for col in columns_dict['beta']:
        print(f"  • {col}")

    print(f"\n PARAMETER COUNTS ({len(columns_dict['param_counts'])} columns):")
    for col in columns_dict['param_counts']:
        print(f"  • {col}")

    print(f"\n BEHAVIORAL MEASURES ({len(columns_dict['behavioral'])} columns):")
    for col in columns_dict['behavioral']:
        print(f"  • {col}")

    print(f"\nSUMMARY:")
    print(f"  Total SINDY columns: {len(columns_dict['all_sindy'])}")
    print(f"  Total modules: {len(modules)}")
    print(f"  Beta columns: {len(columns_dict['beta'])}")
    print(f"  Behavioral columns: {len(columns_dict['behavioral'])}")
    print(f"  Parameter count columns: {len(columns_dict['param_counts'])}")

    # Run analyses
    sindy_module_analysis(df, columns_dict, output_dir)
    corr_df = sindy_behavioral_analysis(df, columns_dict, output_dir)
    sindy_age_analysis(df, columns_dict, output_dir)
    sindy_overview(df, columns_dict, output_dir)
    beta_behavior_age_analysis(df, output_dir)
    scatter_switch_rate_vs_beta(df, output_dir)


if __name__ == "__main__":
    main()
