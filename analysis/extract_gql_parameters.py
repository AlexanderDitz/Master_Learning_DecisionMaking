#!/usr/bin/env python3
"""
Extract GQL Model Parameters for Every Participant
Saves parameters to 'gql_parameters.csv'.
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

# Load GQL model
print("Loading GQL model from:", gql_path)
gql_loaded = load_gql_model(gql_path, deterministic=False)
if not gql_loaded:
    print("Failed to load GQL model.")
    sys.exit(1)
agent_gql_list, _ = gql_loaded

# Load dataset to get participant IDs
dataset = load_dezfouli_dataset()
if dataset is None:
    print("Failed to load dataset.")
    sys.exit(1)
participant_ids = list(dataset.keys())

# Prepare to extract parameters
rows = []
for agent in agent_gql_list:
    # Each agent corresponds to a participant
    pid = getattr(agent, 'participant_id', None)
    if pid is None:
        # Fallback: use order from dataset
        idx = agent_gql_list.index(agent)
        pid = participant_ids[idx] if idx < len(participant_ids) else f"unknown_{idx}"
    # Extract parameters from agent._model (assume torch or numpy arrays)
    params = {}
    if hasattr(agent, '_model'):
        state_dict = agent._model.state_dict()
        for k, v in state_dict.items():
            # Convert tensor/array to flat list
            if hasattr(v, 'cpu'):
                arr = v.cpu().detach().numpy().flatten()
            elif hasattr(v, 'numpy'):
                arr = v.numpy().flatten()
            else:
                arr = np.array(v).flatten()
            # Store each parameter as separate columns if vector, or as scalar
            if arr.size == 1:
                params[k] = arr.item()
            else:
                for i, val in enumerate(arr):
                    params[f"{k}_{i}"] = val
    params['participant'] = pid
    rows.append(params)

# Create DataFrame and save
params_df = pd.DataFrame(rows)
params_df = params_df.set_index('participant')
params_df.to_csv("gql_parameters.csv")
print("✅ Saved GQL parameters for all participants to gql_parameters.csv")
