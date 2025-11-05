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
synthetic_data_path = "data/synthetic_data/dezfouli2019_generated_behavior_rnn_l2_0_001.csv"

print("Loading RNN model from:", rnn_path)
agent_rnn = load_rnn_model(rnn_path, deterministic=False)
if not agent_rnn:
    print("Failed to load RNN model.")
    sys.exit(1)

n_cells = agent_rnn._model.n_cells if hasattr(agent_rnn._model, 'n_cells') else 32
n_actions = agent_rnn._model.n_actions if hasattr(agent_rnn._model, 'n_actions') else 2

# Per-participant extraction using real data
dataset = load_dezfouli_dataset()
if dataset is None:
    print("Failed to load dataset.")
    sys.exit(1)
participant_ids = list(dataset.keys())

rows_participant = []
pid_map = {p: i for i, p in enumerate(participant_ids)}
for pid in participant_ids:
    participant_data = dataset[pid]
    try:
        choices = participant_data['choice'].astype(int).to_numpy()
        choice_onehot = np.eye(n_actions)[choices]
        if f'reward_0' in participant_data.columns and f'reward_1' in participant_data.columns:
            reward_vec = participant_data[[f'reward_0', f'reward_1']].to_numpy()
        else:
            rewards = participant_data['reward'].to_numpy()
            reward_vec = np.full((len(rewards), n_actions), -1, dtype=np.float32)
            for i, (ch, r) in enumerate(zip(choices, rewards)):
                reward_vec[i, ch] = r
        participant_idx = np.full((len(choices), 1), pid_map[pid], dtype=np.float32)
        inputs_np = np.hstack([choice_onehot, reward_vec, participant_idx])
        inputs = torch.tensor(inputs_np, dtype=torch.float32).unsqueeze(1)

        prev_state = None
        for t in range(len(choices)):
            input_t = inputs[t:t+1]
            logits, state = agent_rnn._model(input_t, prev_state=prev_state, batch_first=False)
            prev_state = state
            hidden_vec = state['x_value_reward'].cpu().detach().numpy().flatten()
        # Save only the final hidden state for participant
        row_participant = {f"h_{i}": val for i, val in enumerate(hidden_vec)}
        row_participant['participant'] = pid
        rows_participant.append(row_participant)

    except Exception as e:
        print(f"Error extracting hidden state for participant {pid}: {e}")

hidden_df = pd.DataFrame(rows_participant)
hidden_df = hidden_df.set_index('participant')
hidden_df.to_csv("rnn_hidden_states.csv")
print("✅ Saved RNN hidden states for clustering to rnn_hidden_states.csv")

# Per-trial extraction using synthetic data
synthetic_df = pd.read_csv(synthetic_data_path)
rows_per_trial = []
# Participant ID mapping
participant_ids = synthetic_df['id'].unique()
pid_map = {pid: i for i, pid in enumerate(participant_ids)}
for idx, row in synthetic_df.iterrows():
    choice = int(row['choice'])
    reward = float(row['reward'])
    participant = row['id'] if 'id' in row else None
    session = row['session'] if 'session' in row else None

    participant_idx = pid_map[participant]
    choice_onehot = np.eye(n_actions)[choice]
    reward_vec = np.full(n_actions, -1, dtype=np.float32)
    reward_vec[choice] = reward
    input_np = np.hstack([choice_onehot, reward_vec, [participant_idx]])
    input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    if idx == 0 or (participant is not None and participant != synthetic_df.iloc[idx-1]['id']):
        prev_state = None

    logits, state = agent_rnn._model(
        input_tensor,
        prev_state=prev_state,
        batch_first=False
    )
    prev_state = state

    hidden_vec = state['x_value_reward'].cpu().detach().numpy().flatten()
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
hidden_trial_df.to_csv("rnn_synthetic_data_hidden_states_per_trial.csv", index=False)
print("✅ Saved per-trial RNN hidden states for vector field analysis to rnn_synthetic_data_hidden_states_per_trial.csv")