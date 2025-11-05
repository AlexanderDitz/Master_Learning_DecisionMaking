#!/usr/bin/env python3
"""
Extract LSTM Hidden States for Every Participant and Per Trial
Saves hidden states to 'lstm_hidden_states.csv' (real data, per participant)
and 'lstm_synthetic_data_hidden_states_per_trial.csv' (synthetic data, per trial).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from utils.model_loading_utils import load_lstm_model, load_dezfouli_dataset

# Set up paths
base_params = "params/dezfouli2019"
lstm_path = os.path.join(base_params, "lstm_dezfouli2019.pkl")
synthetic_data_path = "data/synthetic_data/dezfouli2019_generated_behavior_rnn_l2_0_001.csv"

print("Loading LSTM model from:", lstm_path)
agent_lstm = load_lstm_model(lstm_path, deterministic=False)
if not agent_lstm:
    print("Failed to load LSTM model.")
    sys.exit(1)

# --- Per-participant extraction using real data ---
dataset = load_dezfouli_dataset()
if dataset is None:
    print("Failed to load dataset.")
    sys.exit(1)
participant_ids = list(dataset.keys())

n_cells = agent_lstm._model.n_cells if hasattr(agent_lstm._model, 'n_cells') else 32
n_actions = agent_lstm._model.n_actions if hasattr(agent_lstm._model, 'n_actions') else 2

rows = []
for pid in participant_ids:
    participant_data = dataset[pid]
    try:
        # One-hot encode choices
        choices = participant_data['choice'].astype(int).to_numpy()
        choice_onehot = np.eye(n_actions)[choices]  # shape: (seq_len, n_actions)

        # Build reward matrix
        if f'reward_0' in participant_data.columns and f'reward_1' in participant_data.columns:
            reward_vec = participant_data[[f'reward_0', f'reward_1']].to_numpy()
        else:
            rewards = participant_data['reward'].to_numpy()
            reward_vec = np.full((len(rewards), n_actions), -1, dtype=np.float32)
            for i, (ch, r) in enumerate(zip(choices, rewards)):
                reward_vec[i, ch] = r

        # Stack to get LSTM input
        inputs_np = np.hstack([choice_onehot, reward_vec])  # shape: (seq_len, 4)
        inputs = torch.tensor(inputs_np, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, 4)

        # Initial state
        state = (torch.zeros(1, 1, n_cells), torch.zeros(1, 1, n_cells))

        logits, (hidden, cell) = agent_lstm._model(inputs, state)
        hidden_vec = hidden[-1].cpu().detach().numpy().flatten()
    except Exception as e:
        print(f"Error extracting hidden state for participant {pid}: {e}")
        hidden_vec = np.full(n_cells, np.nan)
    row = {f"h_{i}": val for i, val in enumerate(hidden_vec)}
    row['participant'] = pid
    rows.append(row)

hidden_df = pd.DataFrame(rows)
hidden_df = hidden_df.set_index('participant')
hidden_df.to_csv("lstm_hidden_states.csv")
print("✅ Saved LSTM hidden states for all participants to lstm_hidden_states.csv")

# --- Per-trial extraction using synthetic data ---
synthetic_df = pd.read_csv(synthetic_data_path)
participant_ids = synthetic_df['id'].unique()
pid_map = {pid: i for i, pid in enumerate(participant_ids)}

rows_per_trial = []
prev_state = None
prev_participant = None

for idx, row in synthetic_df.iterrows():
    choice = int(row['choice'])
    reward = float(row['reward'])
    participant = row['id']
    session = row['session'] if 'session' in row else None

    participant_idx = pid_map[participant]
    choice_onehot = np.eye(n_actions)[choice]
    reward_vec = np.full(n_actions, -1, dtype=np.float32)
    reward_vec[choice] = reward
    input_np = np.hstack([choice_onehot, reward_vec])
    input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 4)

    # Reset state for new participant
    if prev_participant is None or participant != prev_participant:
        state = (torch.zeros(1, 1, n_cells), torch.zeros(1, 1, n_cells))
    prev_participant = participant

    logits, (hidden, cell) = agent_lstm._model(input_tensor, state)
    state = (hidden, cell)

    hidden_vec = hidden[-1].cpu().detach().numpy().flatten()
    row_trial = {
        'participant': participant,
        'session': session,
        'trial': idx,
        'h_0': hidden_vec[0],
        'h_1': hidden_vec[1],
        'choice': choice,
        'reward': reward
    }
    rows_per_trial.append(row_trial)

hidden_trial_df = pd.DataFrame(rows_per_trial)
hidden_trial_df.to_csv("lstm_synthetic_data_hidden_states_per_trial.csv", index=False)
print("✅ Saved per-trial LSTM hidden states for vector field analysis to lstm_synthetic_data_hidden_states_per_trial.csv")