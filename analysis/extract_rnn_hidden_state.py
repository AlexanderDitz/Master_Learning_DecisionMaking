#!/usr/bin/env python3
"""
Extract RNN Hidden States for Every Participant
Saves hidden states to 'rnn_hidden_states.csv'.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from utils.model_loading_utils import load_rnn_model, load_dezfouli_dataset

# Set up paths
base_params = "params/dezfouli2019"
rnn_path = os.path.join(base_params, "rnn_dezfouli2019_l2_0_001.pkl")

print("Loading RNN model from:", rnn_path)
agent_rnn = load_rnn_model(rnn_path, deterministic=False)
if not agent_rnn:
    print("Failed to load RNN model.")
    sys.exit(1)

# Load dataset to get participant IDs
dataset = load_dezfouli_dataset()
if dataset is None:
    print("Failed to load dataset.")
    sys.exit(1)
participant_ids = list(dataset.keys())

n_cells = agent_rnn._model.n_cells if hasattr(agent_rnn._model, 'n_cells') else 32
n_actions = agent_rnn._model.n_actions if hasattr(agent_rnn._model, 'n_actions') else 2

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

        # Add participant index as feature (repeat for all timesteps)
        participant_idx = np.full((len(choices), 1), int(pid), dtype=np.float32) if pid.isdigit() else np.zeros((len(choices), 1), dtype=np.float32)
        # If pid is not numeric, you may need a mapping from pid to index
        # For now, use a simple mapping:
        pid_map = {p: i for i, p in enumerate(participant_ids)}
        participant_idx = np.full((len(choices), 1), pid_map[pid], dtype=np.float32)

        # Stack to get RNN input
        inputs_np = np.hstack([choice_onehot, reward_vec, participant_idx])  # shape: (seq_len, n_actions*2+1)
        inputs = torch.tensor(inputs_np, dtype=torch.float32).unsqueeze(1)  # shape: (seq_len, 1, features)

        # Initial hidden state (None for your RNN)
        logits, state = agent_rnn._model(inputs, prev_state=None, batch_first=False)
        # Extract hidden state vector (state is a dict)
        hidden_vec = state['x_value_reward'].cpu().detach().numpy().flatten()
    except Exception as e:
        print(f"Error extracting hidden state for participant {pid}: {e}")
        hidden_vec = np.full(n_cells, np.nan)
    row = {f"h_{i}": val for i, val in enumerate(hidden_vec)}
    row['participant'] = pid
    rows.append(row)

hidden_df = pd.DataFrame(rows)
hidden_df = hidden_df.set_index('participant')
hidden_df.to_csv("rnn_hidden_states.csv")
print("✅ Saved RNN hidden states for all participants to rnn_hidden_states.csv")
