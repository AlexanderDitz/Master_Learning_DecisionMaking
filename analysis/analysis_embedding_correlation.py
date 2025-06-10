import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import AgentNetwork, AgentSpice
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from resources.rnn import RLRNN_eckstein2022, RLRNN_dezfouli2019
from resources.sindy_utils import SindyConfig_eckstein2022, SindyConfig_dezfouli2019


path_data = 'data/eckstein2022/eckstein2022_age.csv'
path_rnn = 'params/eckstein2022/rnn_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
path_spice = 'params/eckstein2022/spice_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
demo_cols = ['age']
rnn_class = RLRNN_eckstein2022
sindy_config = SindyConfig_eckstein2022


# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# path_rnn = 'params/dezfouli2019/rnn_dezfouli2019_rldm_l1emb_0_001_l2_0_0001_ep4096.pkl'
# # not trained yet: path_spice = 'params/dezfouli2019/spice_dezfouli2019_rldm_l1emb_0_001_l2_0_0001.pkl'
# demo_cols = ['diag']
# rnn_class = RLRNN_dezfouli2019
# sindy_config = SindyConfig_dezfouli2019


col_participant_id = 'session'
core_cols = [col_participant_id, 'choice', 'reward']

PARTICIPANT_EMB = 0
SINDY_COEFS = 1

# CHANGE THIS VALUE TO SWITCH FROM PARTICIPANT EMBEDDING TO SINDY COEFS
embedding_type = SINDY_COEFS



def calculate_behavioral_metrics(df):
    """
    Calculate behavioral metrics for each participant from bandit data
    """
    behavioral_metrics = []
    
    for participant_id in df[col_participant_id].unique():
        p_data = df[df[col_participant_id] == participant_id].copy()
        # p_data = p_data.sort_values(['experiment_id', 'block']).reset_index(drop=True)
        
        # Basic performance metrics
        avg_reward = p_data['reward'].mean()
        total_trials = len(p_data)
        
        # Calculate stay/switch behavior
        p_data['prev_choice'] = p_data['choice'].shift(1)
        p_data['prev_reward'] = p_data['reward'].shift(1)
        p_data['stayed'] = (p_data['choice'] == p_data['prev_choice'])
        
        # Remove first trial of each block (no previous choice)
        valid_trials = p_data.dropna(subset=['prev_choice', 'prev_reward'])
        
        if len(valid_trials) == 0:
            continue
            
        # Stay probabilities
        stay_after_reward = valid_trials[valid_trials['prev_reward'] == 1]['stayed'].mean()
        stay_after_no_reward = valid_trials[valid_trials['prev_reward'] == 0]['stayed'].mean()
        overall_stay_prob = valid_trials['stayed'].mean()
        
        # Win-stay-lose-shift ratios
        win_stay = stay_after_reward if not pd.isna(stay_after_reward) else 0
        lose_shift = 1 - stay_after_no_reward if not pd.isna(stay_after_no_reward) else 0
        
        # Choice entropy (exploration measure)
        choice_probs = p_data['choice'].value_counts(normalize=True)
        if len(choice_probs) == 2:
            choice_entropy = -np.sum(choice_probs * np.log2(choice_probs))
        else:
            choice_entropy = 0  # All choices were the same
        
        # Learning-related metrics
        # Simple learning proxy: improvement over time
        first_half_reward = p_data.iloc[:len(p_data)//2]['reward'].mean()
        second_half_reward = p_data.iloc[len(p_data)//2:]['reward'].mean()
        learning_improvement = second_half_reward - first_half_reward
        
        # Choice consistency (inverse of switching)
        switch_rate = 1 - overall_stay_prob
        
        behavioral_metrics.append({
            col_participant_id: participant_id,
            'avg_reward': avg_reward,
            'total_trials': total_trials,
            'stay_after_reward': win_stay,
            'stay_after_no_reward': stay_after_no_reward,
            'overall_stay_prob': overall_stay_prob,
            'win_stay_lose_shift': win_stay + lose_shift,
            'choice_entropy': choice_entropy,
            'learning_improvement': learning_improvement,
            'switch_rate': switch_rate
        })
    
    return pd.DataFrame(behavioral_metrics)

def extract_participant_demographics(df, demo_cols):
    """
    Extract demographic/meta information for each participant
    """
    
    if not demo_cols:
        print("No demographic columns found")
        return pd.DataFrame({col_participant_id: df[col_participant_id].unique()})
    
    # map demographics into linear space
    for demo_col in demo_cols:
        unique_values = df[demo_col].unique()
        if isinstance(unique_values[0], str):
            mapping_values = np.linspace(0, 1, len(unique_values))
            mapping = {val: mapping_values[idx] for idx, val in enumerate(unique_values)}
            df[demo_col] = df[demo_col].map(mapping)
    
    # Get first occurrence of each participant (demographics should be constant)
    demographics = df.groupby(col_participant_id)[demo_cols].first().reset_index()
    
    return demographics

def get_embeddings(agent_rnn: AgentNetwork):
    """
    Create dummy participant embeddings for testing
    """
    
    participant_ids = torch.arange(agent_rnn._model.n_participants, dtype=torch.int32)
    embeddings = agent_rnn._model.participant_embedding(participant_ids).detach().numpy()
    
    return embeddings

def get_coefficients(agent_spice: AgentSpice):
    
    coefficients = None
    coefficient_names = []
    for pid in range(agent_spice._model.n_participants):
        agent_spice.new_sess(participant_id=pid)
        
        betas = agent_spice.get_betas()
        modules = agent_spice.get_modules()
        
        if coefficients is None:
            n_coefs = 0
            for beta in betas:
                coefficient_names.append(beta)
            for module in modules:
                n_coefs += max(modules[module][pid].coefficients().shape)
                coefficient_names += [module + ' ' + feature for feature in modules[module][pid].get_feature_names()]
            coefficients = np.zeros((agent_spice._model.n_participants, n_coefs+len(betas)))
        
        for index_beta, beta in enumerate(betas):
            coefficients[pid, index_beta] = betas[beta]
        
        index_coefs = len(betas)
        for module in modules:
            n_coefs_module = max(modules[module][pid].coefficients().shape)
            coefficients[pid, index_coefs:index_coefs+n_coefs_module] = modules[module][pid].coefficients().reshape(-1)
    
    return coefficients, coefficient_names

def perform_embedding_regression(embeddings, behavioral_df, demographic_df, alpha=1.0):
    """
    Perform linear regression of embeddings onto behavioral and demographic variables
    """
    # Merge behavioral and demographic data
    full_data = behavioral_df.merge(demographic_df, on=col_participant_id, how='inner')
    
    # Ensure embeddings match the participants we have data for
    participant_ids = full_data[col_participant_id].values.astype(int)
    embedding_subset = embeddings[participant_ids]  # Assuming participant_id matches index
    
    # Standardize embeddings
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(embedding_subset)
    
    results = {}
    
    # Get target variables (exclude participant_id)
    target_cols = [col for col in full_data.columns if col != col_participant_id]
    
    for target in target_cols:
        y = full_data[target].values
        
        # Skip non-numeric targets
        if not np.issubdtype(y.dtype, np.number):
            print(f"Skipping non-numeric target: {target}")
            continue
            
        # Handle missing values
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < len(y) * 0.5:  # Skip if too many missing
            print(f"Skipping {target} due to too many missing values")
            continue
            
        X_valid = X_scaled[valid_mask]
        y_valid = y[valid_mask]
        
        if len(np.unique(y_valid)) < 2:  # Skip if no variance
            print(f"Skipping {target} due to no variance")
            continue
        
        # Standardize target if continuous
        if len(np.unique(y_valid)) > 10:  # Assume continuous if >10 unique values
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_valid.reshape(-1, 1)).flatten()
        else:
            y_scaled = y_valid
        
        # Fit Ridge regression
        model = Ridge(alpha=alpha)
        model.fit(X_valid, y_scaled)
        
        # Calculate metrics
        y_pred = model.predict(X_valid)
        r2 = r2_score(y_scaled, y_pred)
        
        # Cross-validation R²
        cv_scores = cross_val_score(model, X_valid, y_scaled, cv=5, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()
        
        # Statistical significance test
        # Compare to null model (intercept only)
        null_model = Ridge(alpha=alpha)
        null_model.fit(np.ones((len(X_valid), 1)), y_scaled)
        null_pred = null_model.predict(np.ones((len(X_valid), 1)))
        null_r2 = r2_score(y_scaled, null_pred)
        
        # F-test for overall significance
        n_samples, n_features = X_valid.shape
        f_stat = ((r2 - null_r2) / n_features) / ((1 - r2) / (n_samples - n_features - 1))
        p_value = 1 - stats.f.cdf(f_stat, n_features, n_samples - n_features - 1)
        
        results[target] = {
            'model': model,
            'r2': r2,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'n_samples': len(X_valid),
            'f_statistic': f_stat,
            'p_value': p_value
        }
    
    return results, scaler_X

def plot_regression_results(results):
    """
    Visualize regression results
    """
    # Create summary dataframe
    summary_data = []
    for target, result in results.items():
        summary_data.append({
            'target': target,
            'r2': result['r2'],
            'cv_r2_mean': result['cv_r2_mean'],
            'cv_r2_std': result['cv_r2_std'],
            'p_value': result['p_value'],
            'significant': result['p_value'] < 0.05
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plot R² values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    x_pos = np.arange(len(summary_df))
    axes[0].bar(x_pos - 0.2, summary_df['r2'], 0.4, label='Train R²', alpha=0.7)
    axes[0].errorbar(x_pos + 0.2, summary_df['cv_r2_mean'], 
                     yerr=summary_df['cv_r2_std'], fmt='o', 
                     label='CV R² (mean ± std)', capsize=5)
    axes[0].set_xlabel('Target Variable')
    axes[0].set_ylabel('R²')
    axes[0].set_title('Embedding Regression Performance')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(summary_df['target'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Significance
    colors = ['red' if sig else 'gray' for sig in summary_df['significant']]
    axes[1].bar(x_pos, -np.log10(summary_df['p_value']), color=colors, alpha=0.7)
    axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    axes[1].set_xlabel('Target Variable')
    axes[1].set_ylabel('-log₁₀(p-value)')
    axes[1].set_title('Statistical Significance')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(summary_df['target'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return summary_df

def analyze_coefficient_patterns(results, n_top=3, embedding_names=None):
    """
    Analyze which embedding dimensions are most important across targets
    """
    print("Top embedding dimensions by target:")
    print("=" * 50)
    
    for target, result in results.items():
        coeffs = result['coefficients']
        top_indices = np.argsort(np.abs(coeffs))[-n_top:][::-1]
        
        print(f"\n{target}:")
        print(f"  R² = {result['r2']:.3f}, p = {result['p_value']:.3f}")
        for i, idx in enumerate(top_indices):
            if not embedding_names:
                print(f"  Dim {idx}: {coeffs[idx]:+.3f}")

# Example usage
def main(embedding_type: int = PARTICIPANT_EMB):
    """
    Main analysis pipeline
    """
    # Load your data (replace with actual file path)
    df = pd.read_csv(path_data)
    
    # map participant ids onto arange
    unique_values = df[col_participant_id].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    df[col_participant_id] = df[col_participant_id].map(mapping)
    
    # Calculate behavioral metrics
    print("\nCalculating behavioral metrics...")
    behavioral_df = calculate_behavioral_metrics(df)
    print(f"Calculated metrics for {len(behavioral_df)} participants")
    print("Behavioral metrics:", list(behavioral_df.columns))
    
    # Extract demographics
    print("\nExtracting demographic information...")
    demographic_df = extract_participant_demographics(df, demo_cols=demo_cols)
    print("Demographic variables:", list(demographic_df.columns))
    
    # Create dummy embeddings
    if embedding_type == PARTICIPANT_EMB:
        agent_rnn = setup_agent_rnn(
            class_rnn=rnn_class, 
            path_model=path_rnn,
            list_sindy_signals=sindy_config['rnn_modules']+sindy_config['control_parameters'],
        )
        embeddings = get_embeddings(agent_rnn=agent_rnn)
        embedding_names = None
    else:
        agent_spice = setup_agent_spice(
            class_rnn=rnn_class,
            path_rnn=path_rnn,
            path_data=path_data,
            path_spice=path_spice,
            rnn_modules=sindy_config['rnn_modules'],
            control_parameters=sindy_config['control_parameters'],
            sindy_library_setup=sindy_config['library_setup'],
            sindy_filter_setup=sindy_config['filter_setup'],
            sindy_dataprocessing=sindy_config['dataprocessing_setup'],
            sindy_library_polynomial_degree=1,
        )
        embeddings, embedding_names = get_coefficients(agent_spice=agent_spice)
    
    # Perform regression analysis
    print("\nPerforming embedding regression analysis...")
    results, scaler = perform_embedding_regression(
        embeddings, behavioral_df, demographic_df, alpha=1.0
    )
    
    print(f"\nAnalyzed {len(results)} target variables")
    
    # Display results
    summary_df = plot_regression_results(results)
    analyze_coefficient_patterns(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.round(3))
    
    return df, behavioral_df, demographic_df, embeddings, results

if __name__ == "__main__":
    # Run the analysis
    df, behavioral_df, demographic_df, embeddings, results = main(embedding_type=embedding_type)