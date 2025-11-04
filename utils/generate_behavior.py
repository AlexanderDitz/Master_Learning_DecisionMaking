import sys, os

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import create_dataset, get_update_dynamics, BanditsDrift, BanditsFlip_eckstein2022, Bandits_Standard, Agent, AgentQ
from resources.rnn_utils import DatasetRNN
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.convert_dataset import convert_dataset

# dataset specific SPICE configurations and models
from resources.rnn import RLRNN, RLRNN_eckstein2022, RLRNN_dezfouli2019
from resources.sindy_utils import SindyConfig, SindyConfig_eckstein2022, SindyConfig_dezfouli2019
from benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022

import argparse
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ----------------------- ARGUMENT PARSING ----------------------------
parser = argparse.ArgumentParser(description='Generate synthetic behavior for a given agent type.')
parser.add_argument('--agent_type', type=str, required=True, help='Agent type: rnn, rnn2, rnn3, rnn4, rnn5, lstm, spice, spice2, spice3, spice4, spice5, spice6, benchmark, baseline, q_agent')
args = parser.parse_args()
agent_type = args.agent_type

# ----------------------- GENERAL CONFIGURATION ----------------------------
# agent_type = 'q_agent'  # 'rnn', 'rnn2', 'lstm', 'spice', 'benchmark', 'baseline', 'q_agent'
# n_trials_per_session = 200


# ------------------- CONFIGURATION ECKSTEIN2022 --------------------
# dataset = 'eckstein2022'
# benchmark_model = 'ApAnBrBcfBch'
# baseline_model = 'ApBr'
# class_rnn = RLRNN_eckstein2022
# sindy_config = SindyConfig_eckstein2022
# bandits_environment = BanditsFlip_eckstein2022
# bandits_kwargs_per_session = [
#     {'sigma': 0.2},
#     ]
# n_sessions = 1
# setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
# rl_model = benchmarking_eckstein2022.rl_model
# path_rnn = f'params/{dataset}/rnn_{dataset}_l2_0_0001.pkl'
# path_spice = f'params/{dataset}/spice_{dataset}_l2_0_0001.pkl'
# path_benchmark = f'params/{dataset}/mcmc_{dataset}_BENCHMARK.nc'

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
dataset = 'dezfouli2019'
benchmark_model = 'gql_dezfouli2019_PhiChiBetaKappaC.pkl'
baseline_model = 'PhiBeta'
class_rnn = RLRNN_dezfouli2019
sindy_config = SindyConfig_dezfouli2019
bandits_environment = Bandits_Standard
n_sessions = 12
unique_bandits_kwargs = [
    {'reward_prob_0': 0.25, 'reward_prob_1': 0.05},
    {'reward_prob_0': 0.125, 'reward_prob_1': 0.05},
    {'reward_prob_0': 0.08, 'reward_prob_1': 0.05},
    {'reward_prob_0': 0.05, 'reward_prob_1': 0.25},
    {'reward_prob_0': 0.05, 'reward_prob_1': 0.125},
    {'reward_prob_0': 0.05, 'reward_prob_1': 0.08},
    ]
bandits_kwargs_per_session = unique_bandits_kwargs * 2  # Repeat twice for 12 blocks

setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
Dezfouli2019GQL = benchmarking_dezfouli2019.Dezfouli2019GQL
path_rnn_l2_0_001 = f'params/{dataset}/rnn_{dataset}_l2_0_001.pkl'
path_rnn_l2_0_0001 = f'params/{dataset}/rnn2_{dataset}_l2_0_0001.pkl'
path_rnn_l2_0_00001 = f'params/{dataset}/rnn3_{dataset}_l2_0_00001.pkl'
path_rnn_l2_0_0005 = f'params/{dataset}/rnn4_{dataset}_l2_0_0005.pkl'
path_rnn_l2_0_00005 = f'params/{dataset}/rnn5_{dataset}_l2_0_00005.pkl'
path_spice_l2_0 = f'params/{dataset}/spice_{dataset}_l2_0.pkl'
path_spice_l2_0_001 = f'params/{dataset}/spice2_{dataset}_l2_0_001.pkl'
path_spice_l2_0_0001 = f'params/{dataset}/spice3_{dataset}_l2_0_0001.pkl'
path_spice_l2_0_00001 = f'params/{dataset}/spice4_{dataset}_l2_0_00001.pkl'
path_spice_l2_0_0005 = f'params/{dataset}/spice5_{dataset}_l2_0_0005.pkl'
path_spice_l2_0_00005 = f'params/{dataset}/spice6_{dataset}_l2_0_00005.pkl'
path_benchmark = f'params/{dataset}/gql_{dataset}_PhiChiBetaKappaC.pkl'
path_lstm = f'params/{dataset}/lstm_{dataset}.pkl'

model_paths = {
    'rnn': path_rnn_l2_0_001,
    'rnn2': path_rnn_l2_0_0001,
    'rnn3': path_rnn_l2_0_00001,
    'rnn4': path_rnn_l2_0_0005,
    'rnn5': path_rnn_l2_0_00005,
    'lstm': path_lstm,
    'spice': path_spice_l2_0,
    'spice2': path_spice_l2_0_001,
    'spice3': path_spice_l2_0_0001,
    'spice4': path_spice_l2_0_00001,
    'spice5': path_spice_l2_0_0005,
    'spice6': path_spice_l2_0_00005,
    'benchmark': path_benchmark,
    'baseline': path_benchmark,
    'q_agent': None
}
path_model = model_paths[agent_type]

# ------------------- PIPELINE ----------------------------

# Load real data to get number of trials per participant
real_data_path = f'data/preprocessing/{dataset}.csv'
real_df = pd.read_csv(real_data_path)
trials_per_participant = real_df.groupby('df_participant_id').size().to_dict()
participant_ids = list(trials_per_participant.keys())
n_participants = len(participant_ids)

model_suffix = {
    'rnn': '_l2_0_001',
    'rnn2': '_l2_0_0001',
    'rnn3': '_l2_0_00001',
    'rnn4': '_l2_0_0005',
    'rnn5': '_l2_0_00005',
    'lstm': '',
    'spice': '_l2_0',
    'spice2': '_l2_0_001',
    'spice3': '_l2_0_0001',
    'spice4': '_l2_0_00001',
    'spice5': '_l2_0_0005',
    'spice6': '_l2_0_00005',
    'benchmark': '',
    'baseline': '',
    'q_agent': ''
}
suffix = model_suffix.get(agent_type, '')
path_data = f'data/preprocessing/{dataset}.csv'
# Always use a unique filename for each agent_type
synthetic_data_dir = os.path.join('data', 'synthetic_data')
os.makedirs(synthetic_data_dir, exist_ok=True)
path_save = os.path.join(synthetic_data_dir, f'{dataset}_generated_behavior_{agent_type}{suffix}.csv')
# path_save = f'synthetic_data/{dataset}_generated_behavior_{agent_type}{suffix}.csv'
if agent_type in ['baseline', 'benchmark']:
    path_benchmark = path_benchmark.replace('BENCHMARK', agent_type)

# check if generated behavior file exists
data_files = os.listdir(synthetic_data_dir)
count_files_generated = 0
for f in data_files:
    if os.path.basename(path_save).split('.')[0] in f:
        count_files_generated += 1
if count_files_generated > 0:
    path_save = path_save.split('.')[0] + f'_{count_files_generated}.csv'

from utils.model_loading_utils import load_lstm_model

def get_setup_agent(agent_type):
    if agent_type.startswith('spice'):
        return setup_agent_spice
    elif agent_type.startswith('rnn'):
        return setup_agent_rnn
    elif agent_type == 'lstm':
        return lambda **kwargs: load_lstm_model(path_lstm, deterministic=False)
    elif agent_type in ['baseline', 'benchmark']:
        return setup_agent_benchmark
    elif agent_type == 'q_agent':
        return lambda **kwargs: AgentQ(
            alpha_reward=0.3, 
            beta_reward=3,
            alpha_penalty=0.6,
            alpha_counterfactual_reward=0.3,
            alpha_counterfactual_penalty=0.6,
            beta_choice=1.0,
        )
    else:
        raise ValueError(f'Unknown agent_type: {agent_type}')

setup_agent = get_setup_agent(agent_type)

# Generate synthetic data
print(f'Generating synthetic data for agent_type: {agent_type}')
n_participants = len(participant_ids)
    
dataset_xs, dataset_ys = [], []
meta_rows = []
for i, participant_id in enumerate(participant_ids):
    n_trials = trials_per_participant[participant_id]
    # Optionally, split n_trials across sessions if needed
    trials_per_session = n_trials // n_sessions
    remainder = n_trials % n_sessions

    trial_counter = 0
    for session_idx in range(n_sessions):
        # Distribute remainder trials to the first sessions
        n_trials_this_session = trials_per_session + (1 if session_idx < remainder else 0)
        environment = bandits_environment(
            **bandits_kwargs_per_session[session_idx],
        )
        if agent_type.startswith('spice'):
            spice_rnn_paths = {
                'spice2': path_rnn_l2_0_001,
                'spice3': path_rnn_l2_0_0001,
                'spice4': path_rnn_l2_0_00001,
                'spice5': path_rnn_l2_0_0005,
                'spice6': path_rnn_l2_0_00005,
            }
            agent = setup_agent(
                class_rnn=class_rnn,
                path_rnn=spice_rnn_paths[agent_type],
                path_spice=path_model,
                rnn_modules=sindy_config['rnn_modules'],
                control_parameters=sindy_config['control_parameters'],
                sindy_library_polynomial_degree=1,
                sindy_library_setup=sindy_config['library_setup'],
                sindy_filter_setup=sindy_config['filter_setup'],
                sindy_dataprocessing=sindy_config['dataprocessing_setup'],
                deterministic=False
            )
        elif agent_type.startswith('rnn') or agent_type in ['baseline', 'benchmark']:
            agent = setup_agent(
                class_rnn=class_rnn,
                path_rnn=path_model if agent_type.startswith('rnn') else None,
                path_model=path_model if agent_type in ['baseline', 'benchmark'] else None,
                deterministic=False,
                model_config=benchmark_model if agent_type == 'benchmark' else baseline_model,
        )
        elif agent_type in ['lstm', 'q_agent']:
            agent = setup_agent()
        else:
            raise ValueError(f'agent_type ({agent_type}) is unknown.')

        if isinstance(agent, tuple):
            agent = agent[0]
        dataset, _, _, q_values_all_sessions = create_dataset(
            agent=agent,
            environment=environment,
            n_trials=n_trials_this_session,
            n_sessions=1,  # Only one participant per call
            verbose=False,
        )
        n_actions = agent[0]._n_actions if isinstance(agent, list) else agent._n_actions
        for trial_idx in range(n_trials_this_session):
            experiment = dataset.xs[0][trial_idx].cpu().numpy()
            qvals = q_values_all_sessions[0][trial_idx]  # shape: (n_actions,)
            Q0 = qvals[0]
            Q1 = qvals[1]
            # Print action logits/probs for the first 10 trials of the first participant/session
            if i == 0 and session_idx == 0 and trial_idx < 10:
                print(f"Trial {trial_idx} action logits/probs: {experiment[:n_actions]}")
            meta_rows.append([
                participant_id,  # use real participant id
                agent_type,
                session_idx,
                trial_counter,
                int(np.argmax(experiment[:n_actions])),
                float(np.max(experiment[n_actions:n_actions*2])),
                Q0,
                Q1
            ])
            trial_counter += 1

columns = ['id', 'model_type', 'session', 'n_trials', 'choice', 'reward', 'Q0', 'Q1']
df = pd.DataFrame(meta_rows, columns=columns)
df.to_csv(path_save, index=False)

print(f'Data saved to {path_save}')