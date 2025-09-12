import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# standard methods and classes used for every model evaluation
from benchmarking import benchmarking_dezfouli2019
from resources.model_evaluation import get_scores, log_likelihood, akaike_information_criterion, bayesian_information_criterion
from resources.bandits import get_update_dynamics, AgentQ
from resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.convert_dataset import convert_dataset

# dataset specific SPICE models
from resources import rnn, sindy_utils

# dataset specific benchmarking models
from benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022, benchmarking_lstm
from benchmarking.benchmarking_dezfouli2019 import Dezfouli2019GQL

l2_values = ['0', '0.00001', '0.00005', '0.0001', '0.0005', '0.001']

# -------------------------------------------------------------------------------
# AGENT CONFIGURATIONS
# -------------------------------------------------------------------------------

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
study = 'eckstein2022'
train_test_ratio = 0.8
sindy_config = sindy_utils.SindyConfig_eckstein2022
rnn_class = rnn.RLRNN_eckstein2022
additional_inputs = None
setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
rl_model = benchmarking_eckstein2022.rl_model
model_config_baseline = 'ApBr'
model_config_benchmark = 'ApAnBrBcfBch'
benchmark_file = f'mcmc_eckstein2022_benchmark.nc'
baseline_file = f'mcmc_eckstein2022_baseline.nc'

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
# study = 'dezfouli2019'
# train_test_ratio = [3, 6, 9]
# sindy_config = sindy_utils.SindyConfig_eckstein2022
# rnn_class = rnn.RLRNN_eckstein2022
# additional_inputs = None
# setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
# gql_model = benchmarking_dezfouli2019.Dezfouli2019GQL
# model_config_baseline = 'PhiBeta'
# model_config_benchmark = 'PhiChiBetaKappaC'
# benchmark_file = f'gql_dezfouli2019_benchmark.pkl'
# baseline_file = f'gql_dezfouli2019_baseline.pkl'

# ------------------------- CONFIGURATION FILE PATHS ------------------------
path_data = f'data/{study}/{study}.csv'
path_model_baseline = os.path.join(f'params/{study}/', baseline_file)
path_model_benchmark = os.path.join(f'params/{study}', benchmark_file)
path_model_benchmark_lstm = f'params/{study}/lstm_{study}.pkl'

path_model_rnn = f'params/{study}/rnn_{study}_l2_L2VALUE.pkl'
path_model_spice = f'params/{study}/spice_{study}_l2_L2VALUE.pkl'

# -------------------------------------------------------------------------------
# MODEL COMPARISON PIPELINE
# -------------------------------------------------------------------------------

dataset = convert_dataset(path_data, additional_inputs=additional_inputs)[0]
participant_ids = dataset.xs[:, 0, -1].unique().cpu().numpy()

# ------------------------------------------------------------
# Setup of agents
# ------------------------------------------------------------

# setup baseline model
if path_model_baseline:
    print("Setting up baseline agent from file", path_model_baseline)
    agent_baseline = setup_agent_benchmark(path_model=path_model_baseline, model_config=model_config_baseline)
else:
    print("Setting up dummy baseline agent")
    agent_baseline = [[AgentQ(alpha_reward=0.2, beta_reward=1, beta_choice=3) for _ in range(len(dataset))], 2]
n_parameters_baseline = 2

# setup benchmark model
if path_model_benchmark:
    print("Setting up benchmark agent from file", path_model_benchmark)
    agent_benchmark = setup_agent_benchmark(path_model=path_model_benchmark, model_config=model_config_benchmark)
    n_parameters_benchmark = agent_benchmark[1]
else:
    n_parameters_benchmark = 0

if path_model_benchmark_lstm:
    print("Setting up LSTM agent from file", path_model_benchmark_lstm)
    agent_lstm = benchmarking_lstm.setup_agent_lstm(path_model=path_model_benchmark_lstm)
    n_parameters_lstm = sum(p.numel() for p in agent_lstm._model.parameters() if p.requires_grad)
else:
    n_parameters_lstm = 0
    
# setup rnn agent
if path_model_rnn is not None:
    agent_rnn = {}
    for value in l2_values:
        current_rnn = path_model_rnn.replace('L2VALUE', value.replace('.', '_'))
        print("Setting up RNN agent from file", current_rnn)
        agent_rnn[value] = setup_agent_rnn(
            class_rnn=rnn_class,
            path_rnn=current_rnn,
            )
    n_parameters_rnn = sum(p.numel() for p in agent_rnn[value]._model.parameters() if p.requires_grad) - agent_rnn[value]._model.embedding_size * (len(participant_ids)-1)
else:
    n_parameters_rnn = 0
    
# setup spice agent
if path_model_spice is not None:
    agent_spice = {}
    for value in l2_values:
        current_rnn = path_model_rnn.replace('L2VALUE', value.replace('.', '_'))
        current_spice = path_model_spice.replace('L2VALUE', value.replace('.', '_'))
        print("Setting up SPICE agent from file", current_spice)
        agent_spice[value] = setup_agent_spice(
            class_rnn=rnn_class,
            path_rnn=current_rnn,
            path_spice=current_spice,
        )
n_parameters_spice = 0

# ------------------------------------------------------------
# Dataset splitting
# ------------------------------------------------------------

# split data into train and test data according to train_test_ratio
if isinstance(train_test_ratio, float):
    dataset_train, dataset_test = split_data_along_timedim(dataset, split_ratio=train_test_ratio)
    
elif isinstance(train_test_ratio, list) or isinstance(train_test_ratio, tuple):
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, list_test_sessions=train_test_ratio)
    
else:
    raise TypeError("train_test_raio must be either a float number or a list of integers containing the session/block ids which should be used as test sessions/blocks")

# ------------------------------------------------------------
# Computation of metrics
# ------------------------------------------------------------

print('Running model evaluation...')

# Initialize arrays for both train and test data
n_models = 3 + 2*len(l2_values)  # Baseline, Benchmark, LSTM + RNN and SPICE for each l2_value
table_values_raw_train = np.zeros((n_models, 7, len(dataset_train)))
table_values_raw_test = np.zeros((n_models, 7, len(dataset_test)))
considered_trials_train = np.zeros((n_models, 1, len(dataset_train)))
considered_trials_test = np.zeros((n_models, 1, len(dataset_test)))

failed_attempts = 0

# Main computation loop
for dataset_type, current_dataset, table_raw, considered_trials in [
    ('train', dataset_train, table_values_raw_train, considered_trials_train),
    ('test', dataset_test, table_values_raw_test, considered_trials_test)
]:
    
    print(f'Computing metrics for {dataset_type} data...')
    
    for index_data in tqdm(range(len(current_dataset))):
        try:
            pid = current_dataset.xs[index_data, 0, -1].int().item()
            
            if not pid in participant_ids:
                print(f"Skipping participant {index_data} because they could not be found in the SPICE participants. Probably due to prior filtering of badly fitted participants.")
                continue
            
            # Determine data input based on train_test_ratio type and current dataset type
            if isinstance(train_test_ratio, float):
                # For float ratio: use full dataset for computation but split for evaluation
                data_input = dataset.xs
                data_ys_full = dataset.xs[index_data, :, :agent_baseline[0][0]._n_actions].cpu().numpy()
                
                # Get probabilities for all models using full data
                probs_baseline = get_update_dynamics(experiment=data_input[index_data], agent=agent_baseline[0][pid])[1]
                n_trials = len(probs_baseline)
                
                if dataset_type == 'train':
                    # Training data: first 80% of trials
                    index_start = 0
                    index_end = int(n_trials * train_test_ratio)
                else:
                    # Test data: last 20% of trials (but need training for warm-up)
                    index_start = int(n_trials * train_test_ratio)
                    index_end = n_trials
                    
                data_ys = data_ys_full[index_start:index_end]
                
            else:
                # For list ratio: train and test are separate datasets
                data_input = current_dataset.xs
                probs_baseline = get_update_dynamics(experiment=data_input[index_data], agent=agent_baseline[0][pid])[1]
                n_trials = len(probs_baseline)
                data_ys = current_dataset.xs[index_data, :n_trials, :agent_baseline[0][0]._n_actions].cpu().numpy()
                index_start = 0
                index_end = n_trials
            
            # Baseline model
            scores_baseline = np.array(get_scores(
                data=data_ys, 
                probs=probs_baseline[index_start:index_end], 
                n_parameters=agent_baseline[1]
            ))
            table_raw[0, -3:, index_data] = scores_baseline
            table_raw[0, 0, index_data] = n_parameters_baseline
            considered_trials[0, 0, index_data] += index_end - index_start
            
            # Benchmark model
            if path_model_benchmark:
                if isinstance(train_test_ratio, float):
                    probs_benchmark = get_update_dynamics(experiment=dataset.xs[index_data], agent=agent_benchmark[0][pid])[1]
                else:
                    probs_benchmark = get_update_dynamics(experiment=data_input[index_data], agent=agent_benchmark[0][pid])[1]
                    
                scores_benchmark = np.array(get_scores(
                    data=data_ys, 
                    probs=probs_benchmark[index_start:index_end], 
                    n_parameters=n_parameters_benchmark
                ))
                table_raw[1, -3:, index_data] = scores_benchmark
                table_raw[1, 0, index_data] = n_parameters_benchmark
                considered_trials[1, 0, index_data] += index_end - index_start

            # Benchmark LSTM
            if path_model_benchmark_lstm:
                if isinstance(train_test_ratio, float):
                    probs_lstm = get_update_dynamics(experiment=dataset.xs[index_data], agent=agent_lstm)[1]
                else:
                    probs_lstm = get_update_dynamics(experiment=data_input[index_data], agent=agent_lstm)[1]
                    
                scores_lstm = np.array(get_scores(
                    data=data_ys, 
                    probs=probs_lstm[index_start:index_end], 
                    n_parameters=n_parameters_lstm
                ))
                table_raw[2, -3:, index_data] = scores_lstm
                table_raw[2, 0, index_data] = n_parameters_lstm
                considered_trials[2, 0, index_data] += index_end - index_start
                
            # RNN and SPICE models
            for index_value, value in enumerate(l2_values):
                
                # SPICE-RNN
                if path_model_rnn is not None:
                    if isinstance(train_test_ratio, float):
                        probs_rnn = get_update_dynamics(experiment=dataset.xs[index_data], agent=agent_rnn[value])[1]
                    else:
                        probs_rnn = get_update_dynamics(experiment=data_input[index_data], agent=agent_rnn[value])[1]
                        
                    scores_rnn = np.array(get_scores(
                        data=data_ys, 
                        probs=probs_rnn[index_start:index_end], 
                        n_parameters=n_parameters_rnn
                    ))
                    table_raw[3+2*index_value, -3:, index_data] = scores_rnn
                    table_raw[3+2*index_value, 0, index_data] = n_parameters_rnn
                    considered_trials[3+2*index_value, 0, index_data] += index_end - index_start
                
                # SPICE
                if path_model_spice is not None:
                    if isinstance(train_test_ratio, float):
                        additional_inputs_embedding = dataset.xs[0, agent_spice[value]._n_actions*2:-3]
                        probs_spice = get_update_dynamics(experiment=dataset.xs[index_data], agent=agent_spice[value])[1]
                    else:
                        additional_inputs_embedding = data_input[0, agent_spice[value]._n_actions*2:-3]
                        probs_spice = get_update_dynamics(experiment=data_input[index_data], agent=agent_spice[value])[1]
                    
                    agent_spice[value].new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
                    n_params_spice = agent_spice[value].count_parameters()[pid]
                    
                    scores_spice = np.array(get_scores(
                        data=data_ys, 
                        probs=probs_spice[index_start:index_end], 
                        n_parameters=n_params_spice
                    ))
                    table_raw[3+2*index_value+1, -3:, index_data] = scores_spice
                    table_raw[3+2*index_value+1, 0, index_data] = n_params_spice
                    considered_trials[3+2*index_value+1, 0, index_data] += index_end - index_start
            
        except Exception as e:  
            print(e)
            failed_attempts += 1

# ------------------------------------------------------------
# Post processing for both train and test data
# ------------------------------------------------------------

def process_table_values(table_values_raw, considered_trials):
    """Process raw table values to compute final metrics"""
    table_values = np.zeros(table_values_raw.shape[:-1])
    
    mask_non_zero = table_values_raw[:, -3, 0] != 0
    
    # compute averaged scores and replace with summed scores in table_values (table_values[..., -3:])
    table_values[mask_non_zero, -3:] += table_values_raw[mask_non_zero, -3:].sum(axis=-1) / considered_trials[mask_non_zero, :1].sum(axis=-1)
    # compute average trial likelihood and std from NLL
    table_values[mask_non_zero, 2] += np.exp(-table_values_raw[mask_non_zero, -3].sum(axis=-1) / considered_trials[mask_non_zero, 0].sum(axis=-1))
    table_values[mask_non_zero, 3] += np.exp(-table_values_raw[mask_non_zero, -3] / considered_trials[mask_non_zero, 0]).std(axis=-1)

    # compute mean and std for n_parameters
    table_values[mask_non_zero, 0] += table_values_raw[mask_non_zero, 0].mean(axis=-1)
    table_values[mask_non_zero, 1] += table_values_raw[mask_non_zero, 0].std(axis=-1)
    
    return table_values

table_values_train = process_table_values(table_values_raw_train, considered_trials_train)
table_values_test = process_table_values(table_values_raw_test, considered_trials_test)

# ------------------------------------------------------------
# Generate separate tables for training and test data
# ------------------------------------------------------------

def generate_latex_table(table_values, table_type, study):
    """Generate and save latex table for given data"""
    headers = ['$n_\\text{parameters}$', '$(\sigma)$', '$\\bar{\mathcal{L}}$', '($\sigma$)', 'NLL', 'AIC', 'BIC']
    indexes = ['Baseline', 'Benchmark', 'LSTM']
    for value in l2_values:
        indexes.append(value)
        indexes.append('RNN')
        indexes.append('SPICE')
    
    # Save to CSV
    try:
        df = pd.DataFrame(table_values, columns=headers)
    except Exception:
        df = pd.DataFrame(table_values.T, columns=headers)
    
    os.makedirs('analysis/analysis_model_evaluation_tables', exist_ok=True)
    df.to_csv(os.path.join('analysis/analysis_model_evaluation_tables', f'model_evaluation_{study}_{table_type}.csv'), index=False)
    
    # Print latex table content
    print(f"\n{'='*50}")
    print(f"{table_type.upper()} DATA TABLE")
    print(f"{'='*50}")
    
    str_headers = ""
    for header in headers:
        str_headers += "&" + header
    str_headers += "\\\\"
    print("\\toprule")
    print(str_headers)
    print("\\midrule")

    n_l2_values_printed = 0
    for index_index, index in enumerate(indexes):
        if index in l2_values:
            print("\\midrule")
        str_content = index
        if index in indexes[:3]:
            for index_header, header in enumerate(headers):
                if index_header == 0 or index_header == 1:
                    str_content += "&" + f"{int(table_values[index_index, index_header])}"
                else:
                    str_content += "&" + f"{table_values[index_index, index_header]:.5f}"
            str_content += "\\\\"
            print(str_content)
        elif index in l2_values:
            str_content = f"$l_2={index}$"
            str_content += "&"*len(headers)
            str_content += "\\\\"
            print(str_content)
            n_l2_values_printed += 1
        else:
            for index_header, header in enumerate(headers):
                if index_header == 0 or index_header == 1: 
                    if index == 'SPICE':
                        str_content += "&" + f"{table_values[index_index-n_l2_values_printed, index_header]:.2f}"
                    else:
                        str_content += "&" + f"{int(table_values[index_index-n_l2_values_printed, index_header])}"
                else:
                    str_content += "&" + f"{table_values[index_index-n_l2_values_printed, index_header]:.5f}"
            str_content += "\\\\"
            print(str_content)
    
    print("\\bottomrule")

# Generate both tables
generate_latex_table(table_values_train, 'train', study)
generate_latex_table(table_values_test, 'test', study)

print(f"\nFailed attempts: {failed_attempts}")
print(f"Train/test split type: {'float ({})'.format(train_test_ratio) if isinstance(train_test_ratio, float) else 'list (test blocks: {})'.format(train_test_ratio)}")