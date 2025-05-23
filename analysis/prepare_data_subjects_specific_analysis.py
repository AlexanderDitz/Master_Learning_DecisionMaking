import sys
import os
import logging
import numpy as np
import torch
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resources.model_evaluation import bayesian_information_criterion, log_likelihood
from utils.plotting import plot_session
from resources.bandits import AgentQ, AgentNetwork, AgentSpice, get_update_dynamics
from resources.rnn import RLRNN
from resources.rnn_utils import DatasetRNN
from resources.rnn_training import fit_model
from resources.sindy_training import fit_spice as fit_model_sindy

np.random.seed(42)
torch.manual_seed(42)

n_actions = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

data_path = 'data/parameter_recovery/data_16p_0.csv'
df = pd.read_csv(data_path)

behavior_metrics = []

for pid in df['session'].unique():
    participant_df = df[df['session'] == pid]
    if participant_df.empty:
        continue

    choices = participant_df['choice'].values
    n_switches = np.sum(np.abs(np.diff(choices)))
    switch_rate = n_switches / (len(choices) - 1) if len(choices) > 1 else 0

    stay_after_reward_count = 0
    stay_after_reward_total = 0
    for i in range(len(choices) - 1):
        current_choice = choices[i]
        next_choice = choices[i+1]
        current_reward = participant_df['reward'].iloc[i]
        if current_reward > 0:
            stay_after_reward_total += 1
            if next_choice == current_choice:
                stay_after_reward_count += 1
    stay_after_reward_rate = stay_after_reward_count / stay_after_reward_total if stay_after_reward_total > 0 else np.nan

    perseveration = np.mean(choices[:-1] == choices[1:]) if len(choices) > 1 else np.nan
    avg_reward = participant_df['reward'].mean()

    behavior_metrics.append({
        'participant_id': pid,
        'switch_rate': switch_rate,
        'stay_after_reward': stay_after_reward_rate,
        'perseveration': perseveration,
        'avg_reward': avg_reward
    })

behavior_df = pd.DataFrame(behavior_metrics)
unique_participants = df['session'].unique()
n_participants = len(unique_participants)

all_xs = []
all_ys = []

participant_index_to_id = {}
participant_params = {}

for i, participant_id in enumerate(unique_participants):
    participant_df = df[df['session'] == participant_id]
    n_trials = len(participant_df)

    alpha_reward = participant_df['alpha_reward'].iloc[0]
    alpha_penalty = participant_df['alpha_penalty'].iloc[0]
    beta_reward = participant_df['beta_reward'].iloc[0]
    beta_choice = participant_df['beta_choice'].iloc[0]
    forget_rate = participant_df['forget_rate'].iloc[0]

    participant_index_to_id[i] = participant_id
    participant_params[participant_id] = {
        'alpha_reward': alpha_reward,
        'alpha_penalty': alpha_penalty,
        'beta_reward': beta_reward,
        'beta_choice': beta_choice,
        'forget_rate': forget_rate
    }

    xs = torch.zeros((1, n_trials, 5))
    for t in range(1, n_trials):
        prev_choice = participant_df['choice'].iloc[t - 1]
        xs[0, t, int(prev_choice)] = 1.0
        if int(prev_choice) == 0:
            xs[0, t, 2] = participant_df['reward'].iloc[t - 1]
            xs[0, t, 3] = -1
        else:
            xs[0, t, 2] = -1
            xs[0, t, 3] = participant_df['reward'].iloc[t - 1]
    xs[0, :, 4] = i

    ys = torch.zeros((1, n_trials, n_actions))
    for t in range(n_trials):
        choice = participant_df['choice'].iloc[t]
        ys[0, t, int(choice)] = 1.0

    all_xs.append(xs)
    all_ys.append(ys)

combined_xs = torch.cat(all_xs)
combined_ys = torch.cat(all_ys)
combined_dataset = DatasetRNN(combined_xs, combined_ys)

list_rnn_modules = [
    'x_learning_rate_reward',
    'x_value_reward_not_chosen',
    'x_value_choice_chosen',
    'x_value_choice_not_chosen'
]

list_control_parameters = ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice']

library_setup = {
    'x_learning_rate_reward': ['c_reward', 'c_value_reward', 'c_value_choice'],
    'x_value_reward_not_chosen': ['c_value_choice'],
    'x_value_choice_chosen': ['c_value_reward'],
    'x_value_choice_not_chosen': ['c_value_reward'],
}

filter_setup = {
    'x_learning_rate_reward': ['c_action', 1, True],
    'x_value_reward_not_chosen': ['c_action', 0, True],
    'x_value_choice_chosen': ['c_action', 1, True],
    'x_value_choice_not_chosen': ['c_action', 0, True],
}

embedding_size = 9
model_rnn = RLRNN(
    n_actions=n_actions,
    n_participants=n_participants,
    list_signals=list_rnn_modules + list_control_parameters,
    hidden_size=22,
    embedding_size=embedding_size,
    dropout=0.319
)

optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=5e-3)
model_rnn, optimizer_rnn, final_train_loss = fit_model(
    model=model_rnn,
    optimizer=optimizer_rnn,
    dataset_train=combined_dataset,
    epochs=1,
    n_steps=62,
    scheduler=False,
    convergence_threshold=0,
    bagging=True
)

agent_rnn = AgentNetwork(model_rnn=model_rnn, n_actions=n_actions)

agent_sindy, _ = fit_model_sindy(
    rnn_modules=list_rnn_modules,
    control_signals=list_control_parameters,
    agent_rnn=agent_rnn,
    data=combined_dataset,
    n_sessions_off_policy=1,
    polynomial_degree=2,
    library_setup=library_setup,
    filter_setup=filter_setup,
    optimizer_threshold=0.05,
    optimizer_alpha=1,
    verbose=True,
)

# Mapping to support beta lookups in count_parameters
mapping_modules_values = {
    'x_value_choice_chosen': 'x_value_choice',
    'x_value_choice_not_chosen': 'x_value_choice',
    'x_learning_rate_reward': 'x_value_reward',
    'x_value_reward_not_chosen': 'x_value_reward',
}
sindy_params = agent_sindy.count_parameters(mapping_modules_values=mapping_modules_values)

# Calculate BIC and log-likelihood values
available_participant_ids = set(range(n_participants))

bic_values = []
ll_values = []

# Process each participant individually 
for pid in sorted(available_participant_ids):
    # Find indices of data belonging to this participant in the original dataset
    indices = []
    for i in range(combined_dataset.xs.shape[0]):
        # Use item() to get scalar value from tensor
        if combined_dataset.xs[i, 0, -1].item() == pid:
            indices.append(i)
    
    if not indices:
        logger.warning(f"No data found for participant {pid}, skipping")
        continue
    
    logger.info(f"Processing participant {pid}, found {len(indices)} data sequences")
    
    # Initialize data collectors for this participant
    all_choices = []
    all_probs = []
    
    # Process each sequence individually to maintain shape consistency
    for idx in indices:
        # Initialize the agent for this participant
        agent_sindy.new_sess(participant_id=pid)
        
        # Get the original tensor directly from the dataset
        # This ensures we use exactly the same tensor shape/format as during training
        x_tensor = combined_dataset.xs[idx].clone() 
        
        try:
            # the original get_update_dynamics function from bandits.py
            _, probs, _ = get_update_dynamics(x_tensor.cpu(), agent_sindy)
            
            # Extract choices for log-likelihood calculation
            choices = x_tensor[..., :agent_sindy._n_actions].cpu().numpy()
            
            # Store results for this sequence
            all_choices.append(choices)
            all_probs.append(probs)
        except Exception as e:
            logger.error(f"Error processing sequence {idx} for participant {pid}: {str(e)}")
            continue
    
    # Combine all sequences for this participant
    if all_choices and all_probs:
        combined_choices = np.vstack(all_choices)
        combined_probs = np.vstack(all_probs)
        
        # Calculate log-likelihood and BIC
        ll = log_likelihood(data=combined_choices, probs=combined_probs)
        n_trials = combined_choices.shape[0]
        normalized_ll = ll / n_trials if n_trials > 0 else 0
        ll_values.append(normalized_ll)
        
        bic = bayesian_information_criterion(
            data=combined_choices, 
            probs=combined_probs, 
            n_parameters=sindy_params[pid], 
            ll=ll
        )
        normalized_bic = bic / n_trials if n_trials > 0 else 0
        bic_values.append(normalized_bic)
    else:
        logger.warning(f"No valid data processed for participant {pid}")
        # Add placeholder values to maintain participant order
        ll_values.append(0)
        bic_values.append(0)

avg_bic = np.mean(bic_values) if bic_values else 0
avg_ll = np.mean(ll_values) if ll_values else 0

sindy_modules = agent_sindy._model.submodules_sindy

sindy_recovered_params = {}

for pid in sorted(available_participant_ids):
    original_pid = participant_index_to_id[pid]
    
    param_data = {
        'total_nonzero': 0,
        'parameters_by_module': {}
    }
    
    # Go through each SINDy module
    for module_name in list_rnn_modules:
        # Get the SINDy model for this module and participant
        sindy_model = sindy_modules[module_name][pid]
        
        coefs = sindy_model.model.steps[-1][1].coef_.flatten()
        
        feature_names = sindy_model.get_feature_names()
        
        # Find non-zero parameters
        nonzero_indices = np.where(np.abs(coefs) > 1e-10)[0]
        nonzero_count = len(nonzero_indices)
        
        # Store non-zero parameters and their values
        nonzero_params = {}
        for idx in nonzero_indices:
            param_name = feature_names[idx]
            param_value = coefs[idx]
            nonzero_params[param_name] = param_value
        
        param_data['total_nonzero'] += nonzero_count
        
        param_data['parameters_by_module'][module_name] = {
            'nonzero_count': nonzero_count,
            'nonzero_params': nonzero_params
        }
    
    sindy_recovered_params[original_pid] = param_data

# Table of most common parameters and their frequencies
most_common_params = {module: {} for module in list_rnn_modules}
for pid, data in sindy_recovered_params.items():
    for module in list_rnn_modules:
        for param_name in data['parameters_by_module'][module]['nonzero_params']:
            if param_name not in most_common_params[module]:
                most_common_params[module][param_name] = 0
            most_common_params[module][param_name] += 1

for module in list_rnn_modules:
    if most_common_params[module]:
        sorted_params = sorted(most_common_params[module].items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\nModule: {module}")
        for param_name, count in sorted_params[:5]:  
            percentage = (count / n_participants) * 100
            logger.info(f"  {param_name}: {count} participants ({percentage:.1f}%)")

# Tables showing parameter values for each module across participants
# For each module, create a DataFrame with parameters as columns and participants as rows
for module_name in list_rnn_modules:
    # Collect all unique parameters found for this module
    all_params = set()
    for pid_data in sindy_recovered_params.values():
        all_params.update(pid_data['parameters_by_module'][module_name]['nonzero_params'].keys())
    
    # Sort parameters for consistent ordering (constants first)
    all_params = sorted(list(all_params), key=lambda x: (0 if x == '1' else 1, x))
    
    # One row per participant, one column per parameter
    module_data = []
    for pid, pid_data in sindy_recovered_params.items():
        row = {'participant_id': pid}
        param_dict = pid_data['parameters_by_module'][module_name]['nonzero_params']
        
        # Fill in parameter values (or 0 if not present for this participant)
        for param in all_params:
            row[param] = param_dict.get(param, 0)
            
        module_data.append(row)
    
    df_module = pd.DataFrame(module_data)
    csv_filename = f'sindy_{module_name}_parameters.csv'
    df_module.to_csv(csv_filename, index=False)
    
    param_stats = {
        'mean': df_module[all_params].mean(),
        'std': df_module[all_params].std(),
        'min': df_module[all_params].min(),
        'max': df_module[all_params].max(),
        'present_in': df_module[all_params].astype(bool).sum() / len(df_module) * 100  # % of participants
    }
    
    stats_filename = f'sindy_{module_name}_param_stats.csv'
    pd.DataFrame(param_stats).to_csv(stats_filename)

# Summary table showing parameter counts by module for all participants
summary_data = []
for pid, pid_data in sindy_recovered_params.items():
    row = {'participant_id': pid}
    for module_name in list_rnn_modules:
        row[f'{module_name}_params'] = pid_data['parameters_by_module'][module_name]['nonzero_count']
    row['total_params'] = pid_data['total_nonzero']
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
summary_filename = 'sindy_parameter_summary.csv'
df_summary.to_csv(summary_filename, index=False)

# Participant embeddings
participant_embeddings = {}
embedding_data = []

for i in range(n_participants):
    pid_tensor = torch.tensor([i], device=agent_rnn._model.device)
    embedding = agent_rnn._model.participant_embedding(pid_tensor).detach().cpu().numpy()[0]
    
    original_pid = participant_index_to_id[i]
    participant_embeddings[original_pid] = embedding
    embedding_data.append(embedding)

embedding_data = np.array(embedding_data)

participant_data = []
for i, pid in enumerate(sorted(participant_params.keys())):
    if i >= len(bic_values):
        continue
        
    params = participant_params[pid]
    embedding = participant_embeddings[pid]
    
    row = {
        'participant_id': pid,
        'bic': bic_values[i],
        'log_likelihood': ll_values[i],
    }
    
    # Add embedding dimensions
    for j in range(embedding_size):
        row[f'embedding_{j}'] = embedding[j]
    
    # Add SINDy parameter counts
    if pid in sindy_recovered_params:
        row['sindy_param_count'] = sindy_recovered_params[pid]['total_nonzero']
        
        # Add counts by module
        for module_name in list_rnn_modules:
            module_count = sindy_recovered_params[pid]['parameters_by_module'][module_name]['nonzero_count']
            row[f'params_{module_name}'] = module_count
    
    participant_data.append(row)

df_analysis = pd.DataFrame(participant_data)
df_analysis.to_csv('embedding_analysis_results.csv', index=False)

# Dataframe with participant parameters and their SINDy parameter counts
final_data = []
for pid in sorted(participant_params.keys()):
    if pid in sindy_recovered_params:
        row = {'participant_id': pid}
        
        # Add original parameters
        row.update(participant_params[pid])
        
        # Add BIC and log-likelihood
        i = list(sorted(participant_params.keys())).index(pid)
        if i < len(bic_values):
            row['bic'] = bic_values[i]
            row['log_likelihood'] = ll_values[i]
        
        # Add parameter counts by module
        for module_name in list_rnn_modules:
            module_count = sindy_recovered_params[pid]['parameters_by_module'][module_name]['nonzero_count']
            row[f'params_{module_name}'] = module_count
            
            # Add specific parameter values
            params = sindy_recovered_params[pid]['parameters_by_module'][module_name]['nonzero_params']
            for param_name, param_value in params.items():
                row[f'{module_name}_{param_name}'] = param_value
        
        row['total_params'] = sindy_recovered_params[pid]['total_nonzero']
        
        # Add embeddings
        if pid in participant_embeddings:
            embedding = participant_embeddings[pid]
            for j in range(embedding_size):
                row[f'embedding_{j}'] = embedding[j]

            # Extract beta values from RNN (they are derived from embeddings)
            pid_tensor = torch.tensor([list(participant_index_to_id.keys())[list(participant_index_to_id.values()).index(pid)]], 
                                    device=agent_rnn._model.device)
            
            # Get beta_reward from the model
            beta_reward = agent_rnn._model.betas['x_value_reward'](
                agent_rnn._model.participant_embedding(pid_tensor)
            ).item()
            
            # Get beta_choice from the model
            beta_choice = agent_rnn._model.betas['x_value_choice'](
                agent_rnn._model.participant_embedding(pid_tensor)
            ).item()
            
            # Add derived beta values to the results
            row['derived_beta_reward'] = beta_reward
            row['derived_beta_choice'] = beta_choice
        
        final_data.append(row)

df_final = pd.DataFrame(final_data)
final_filename = 'sindy_parameter_analysis.csv'
df_final.to_csv(final_filename, index=False)

df_final_behav = pd.merge(df_final, behavior_df, on='participant_id', how='left')
final_filename_behav = 'AAAA_sindy_params_behav_embeddings.csv'
df_final_behav.to_csv(final_filename_behav, index=False)

logger.info("Analysis completed successfully!")