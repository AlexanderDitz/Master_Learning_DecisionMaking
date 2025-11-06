#!/usr/bin/env python3
"""
Extract GQL Model Parameters for Every Participant
Saves parameters to 'gql_parameters.csv' (real data, per participant)
and per-trial Q-values to 'gql_synthetic_data_qvalues_per_trial.csv' (synthetic data, per trial).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import pandas as pd
import numpy as np
from benchmarking.benchmarking_dezfouli2019 import Dezfouli2019GQL, AgentGQL
from utils.model_loading_utils import load_gql_model, load_dezfouli_dataset

# Set up paths
base_params = "params/dezfouli2019"
gql_path = os.path.join(base_params, "gql_dezfouli2019_PhiChiBetaKappaC.pkl")
synthetic_data_path = "data/synthetic_data/dezfouli2019_generated_behavior_benchmark.csv"

# Load GQL model
print("Loading GQL model from:", gql_path)
gql_loaded = load_gql_model(gql_path, deterministic=False)
if not gql_loaded:
    print("Failed to load GQL model.")
    sys.exit(1)
agent_gql_list, _ = gql_loaded

# --- Per-participant parameter extraction using real data ---
dataset = load_dezfouli_dataset()
if dataset is None:
    print("Failed to load dataset.")
    sys.exit(1)
participant_ids = list(dataset.keys())

rows = []
for agent in agent_gql_list:
    pid = getattr(agent, 'participant_id', None)
    if pid is None:
        idx = agent_gql_list.index(agent)
        pid = participant_ids[idx] if idx < len(participant_ids) else f"unknown_{idx}"
    params = {}
    if hasattr(agent, '_model'):
        state_dict = agent._model.state_dict()
        for k, v in state_dict.items():
            if hasattr(v, 'cpu'):
                arr = v.cpu().detach().numpy().flatten()
            elif hasattr(v, 'numpy'):
                arr = v.numpy().flatten()
            else:
                arr = np.array(v).flatten()
            if arr.size == 1:
                params[k] = arr.item()
            else:
                for i, val in enumerate(arr):
                    params[f"{k}_{i}"] = val
    params['participant'] = pid
    rows.append(params)

params_df = pd.DataFrame(rows)
params_df = params_df.set_index('participant')
params_df.to_csv("gql_parameters.csv")
print("✅ Saved GQL parameters for all participants to gql_parameters.csv")

# --- Per-trial Q-value extraction using synthetic data ---
# Load the synthetic data
synthetic_df = pd.read_csv(synthetic_data_path)

# Only keep relevant columns for output
cols_to_save = ['id', 'session', 'n_trials', 'choice', 'reward', 'Q0', 'Q1']

# Save per-trial Q-values for vector field analysis
synthetic_df[cols_to_save].to_csv("gql_synthetic_data_qvalues_per_trial.csv", index=False)
print("✅ Saved per-trial GQL Q-values for vector field analysis to gql_synthetic_data_qvalues_per_trial.csv")