import pandas as pd
import numpy as np
from tqdm import tqdm


import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.convert_dataset import convert_dataset
from resources.rnn_utils import DatasetRNN
from resources.model_evaluation import log_likelihood, bayesian_information_criterion
from resources.bandits import get_update_dynamics, AgentSpice
from resources.sindy_utils import load_spice
from utils.setup_agents import setup_agent_rnn


import pandas as pd
import numpy as np
from tqdm import tqdm

#  BEHAVIORAL METRICS 
data_path = 'data/eckstein2022/eckstein2022.csv'
slcn_path = '/Users/martynaplomecka/closedloop_rl/data/eckstein2022/SLCN.csv'

# (1a) Read raw CSV and cast 'session' → int
original_df = pd.read_csv(data_path)
original_df['session'] = original_df['session'].astype(int)

# (1b) Read SLCN metadata and cast 'ID' → int
slcn_df = pd.read_csv(slcn_path)
slcn_df['ID'] = slcn_df['ID'].astype(int)

# Keep only the columns we care about from SLCN
columns_to_keep = ['ID', 'age - years', 'Category']
available_columns = [col for col in columns_to_keep if col in slcn_df.columns]
slcn_df = slcn_df[available_columns]

# Build a mapping { participant_id → { 'ID', 'age - years', 'Category' } }
slcn_mapping = {}
for _, row in slcn_df.iterrows():
    slcn_mapping[row['ID']] = row.to_dict()

unique_sessions = original_df['session'].unique().tolist()

behavior_metrics = []
for pid in tqdm(unique_sessions, desc="Computing behavior metrics"):
    if pid not in slcn_mapping:
        continue

    participant_df = original_df[original_df['session'] == pid]
    if participant_df.empty:
        continue

    choices = participant_df['choice'].values
    rewards = participant_df['reward'].values

    # — Stay after reward rate
    stay_after_reward_count = 0
    stay_after_reward_total = 0
    for i in range(len(choices) - 1):
        if rewards[i] > 0:
            stay_after_reward_total += 1
            if choices[i + 1] == choices[i]:
                stay_after_reward_count += 1
    stay_after_reward_rate = (
        stay_after_reward_count / stay_after_reward_total
        if stay_after_reward_total > 0
        else 0
    )

    # — Overall switch rate
    switch_count = 0
    switch_total = 0
    for i in range(len(choices) - 1):
        switch_total += 1
        if choices[i + 1] != choices[i]:
            switch_count += 1
    switch_rate = switch_count / switch_total if switch_total > 0 else 0

    # — Perseveration: stay after 3 consecutive unrewarded trials
    perseveration_count = 0
    perseveration_total = 0
    for i in range(3, len(choices)):
        prev_3_rewards = rewards[i - 3 : i]
        prev_3_choices = choices[i - 3 : i]
        if (
            np.all(prev_3_rewards == 0)
            and np.all(prev_3_choices == prev_3_choices[0])
            and len(np.unique(prev_3_choices)) == 1
        ):
            perseveration_total += 1
            if choices[i] == prev_3_choices[0]:
                perseveration_count += 1
    perseveration = (
        perseveration_count / perseveration_total if perseveration_total > 0 else 0
    )

    # — Stay based on last two outcomes ( ++, +−, −+, −− )
    stay_pp = stay_pm = stay_mp = stay_mm = 0
    total_pp = total_pm = total_mp = total_mm = 0
    for i in range(2, len(choices) - 1):
        prev1 = rewards[i - 2]
        prev2 = rewards[i - 1]
        curr = choices[i]
        nxt = choices[i + 1]

        if prev1 > 0 and prev2 > 0:
            total_pp += 1
            if nxt == curr:
                stay_pp += 1
        elif prev1 > 0 and prev2 == 0:
            total_pm += 1
            if nxt == curr:
                stay_pm += 1
        elif prev1 == 0 and prev2 > 0:
            total_mp += 1
            if nxt == curr:
                stay_mp += 1
        elif prev1 == 0 and prev2 == 0:
            total_mm += 1
            if nxt == curr:
                stay_mm += 1

    stay_pp_rate = stay_pp / total_pp if total_pp > 0 else np.nan
    stay_pm_rate = stay_pm / total_pm if total_pm > 0 else np.nan
    stay_mp_rate = stay_mp / total_mp if total_mp > 0 else np.nan
    stay_mm_rate = stay_mm / total_mm if total_mm > 0 else np.nan

    # — Average reward and reaction time
    avg_reward = participant_df['reward'].mean()
    avg_rt = participant_df['rt'].mean()

    sl = slcn_mapping[pid]
    participant_data = {
        'participant_id': pid,
        'stay_after_reward': stay_after_reward_rate,
        'switch_rate': switch_rate,
        'perseveration': perseveration,
        'stay_after_plus_plus': stay_pp_rate,
        'stay_after_plus_minus': stay_pm_rate,
        'stay_after_minus_plus': stay_mp_rate,
        'stay_after_minus_minus': stay_mm_rate,
        'avg_reward': avg_reward,
        'avg_rt': avg_rt,
        'n_trials': len(participant_df),
        'Age': sl.get('age - years', np.nan),
        'Age_Category': sl.get('Category', np.nan)
    }

    behavior_metrics.append(participant_data)

behavior_df = pd.DataFrame(behavior_metrics)
print(f"Behavioral metrics computed for {len(behavior_df)} participants.")



# here we have the SINDy and RNN models part


model_rnn_path = '/Users/martynaplomecka/closedloop_rl/params/eckstein2022/rnn_eckstein2022_reward.pkl'
model_spice_path = '/Users/martynaplomecka/closedloop_rl/params/eckstein2022/spice_eckstein2022_reward.pkl'

agent_rnn = setup_agent_rnn(
    path_model=model_rnn_path,
    list_sindy_signals=[
        'x_learning_rate_reward',
        'x_value_reward_not_chosen',
        'x_value_choice_chosen',
        'x_value_choice_not_chosen',
        'c_action',
        'c_reward_chosen',
        'c_value_reward',
        'c_value_choice'
    ]
)

spice_modules = load_spice(file=model_spice_path)
agent_spice = AgentSpice(
    model_rnn=agent_rnn._model,
    sindy_modules=spice_modules,
    n_actions=agent_rnn._n_actions
)

list_rnn_modules = [
    'x_learning_rate_reward',
    'x_value_reward_not_chosen',
    'x_value_choice_chosen',
    'x_value_choice_not_chosen'
]

n_participants = agent_spice._model.participant_embedding.weight.data.shape[0]
participant_ids = list(range(n_participants))

# Map SPICE index → real pid (from unique_sessions)
index_to_pid = {}
for i, pid in enumerate(unique_sessions):
    if i < n_participants:
        index_to_pid[i] = pid

print(f"Total unique sessions: {len(unique_sessions)}")
print(f"SPICE knows about {n_participants} participants")
print(f"Mapped {len(index_to_pid)} indices to actual PIDs")

# Collect all SINDy feature names
all_feature_names = set()
for module in list_rnn_modules:
    for idx in agent_spice._model.submodules_sindy[module]:
        sindy_model = agent_spice._model.submodules_sindy[module][idx]
        for name in sindy_model.get_feature_names():
            all_feature_names.add(f"{module}_{name}")

# Extract embedding matrix
embedding_matrix = agent_spice._model.participant_embedding.weight.detach().cpu().numpy()
embedding_size = embedding_matrix.shape[1]

# Precompute betas
features = {'beta_reward': {}, 'beta_choice': {}}
for idx in participant_ids:
    if idx not in index_to_pid:
        continue
    agent_spice.new_sess(participant_id=idx)
    betas = agent_spice.get_betas()
    features['beta_reward'][idx] = betas.get('x_value_reward', 0.0)
    features['beta_choice'][idx] = betas.get('x_value_choice', 0.0)

# Build the SINDy‐params DataFrame
sindy_params = []
for idx in tqdm(participant_ids, desc="Extracting SINDy/RNN params"):
    if idx not in index_to_pid:
        continue
    pid = index_to_pid[idx]
    param_dict = {'participant_id': pid}

    for feat in all_feature_names:
        param_dict[feat] = 0.0

    # Insert beta_reward and beta_choice
    param_dict['beta_reward'] = features['beta_reward'].get(idx, 0.0)
    param_dict['beta_choice'] = features['beta_choice'].get(idx, 0.0)

    # Fill in each submodule’s coefficients
    for module in list_rnn_modules:
        if idx in agent_spice._model.submodules_sindy[module]:
            model = agent_spice._model.submodules_sindy[module][idx]
            coefs = model.model.steps[-1][1].coef_.flatten()
            for i, name in enumerate(model.get_feature_names()):
                param_dict[f"{module}_{name}"] = coefs[i]
            param_dict[f"params_{module}"] = np.sum(np.abs(coefs) > 1e-10)
        else:
            param_dict[f"params_{module}"] = 0

    # Total nonzero SINDy parameters across all four submodules
    param_dict['total_params'] = sum(param_dict[f"params_{m}"] for m in list_rnn_modules)

    # Append embedding entries
    if idx < embedding_matrix.shape[0]:
        for j in range(embedding_size):
            param_dict[f'embedding_{j}'] = embedding_matrix[idx, j]
    else:
        for j in range(embedding_size):
            param_dict[f'embedding_{j}'] = np.nan

    sindy_params.append(param_dict)

sindy_df = pd.DataFrame(sindy_params)
print(f"Number of participants in SINDY DataFrame: {len(sindy_df)}")
print(f"Number of participants in behavior DataFrame: {len(behavior_df)}")

# ─── MERGE 1: ONLY KEEP INTERSECTION OF SINDY & BEHAVIOR ───────────────────────────────────────────────

merged_df = pd.merge(sindy_df, behavior_df, on='participant_id', how='inner')
print(f"After inner‐join of SINDy + behavior: {len(merged_df)} participants")

# ─── SECTION 3: CALCULATE MODEL EVALUATION METRICS ─────────────────────────────────────────────────────

dataset_test, _, _, _ = convert_dataset(data_path)

metrics_data = []
for idx in tqdm(participant_ids, desc="Computing model metrics"):
    if idx not in index_to_pid:
        continue
    real_pid = index_to_pid[idx]

    # Only compute metrics if real_pid was in the merged_df
    if real_pid not in set(merged_df['participant_id']):
        continue

    mask = (dataset_test.xs[:, 0, -1] == idx)
    if not mask.any():
        continue

    participant_data = DatasetRNN(*dataset_test[mask])

    # Reset agents
    agent_spice.new_sess(participant_id=idx)
    agent_rnn.new_sess(participant_id=idx)

    # Get predicted probabilities
    _, probs_spice, _ = get_update_dynamics(
        experiment=participant_data.xs, agent=agent_spice
    )
    _, probs_rnn, _ = get_update_dynamics(
        experiment=participant_data.xs, agent=agent_rnn
    )
    n_trials_test = len(probs_spice)
    if n_trials_test == 0:
        continue

    true_actions = participant_data.ys[0, :n_trials_test].cpu().numpy()
    ll_spice = log_likelihood(data=true_actions, probs=probs_spice)
    ll_rnn = log_likelihood(data=true_actions, probs=probs_rnn)

    spice_per_trial_like = np.exp(ll_spice / (n_trials_test * agent_rnn._n_actions))
    rnn_per_trial_like = np.exp(ll_rnn / (n_trials_test * agent_rnn._n_actions))

    n_params_dict = agent_spice.count_parameters(
        mapping_modules_values={
            m: 'x_value_choice' if 'choice' in m else 'x_value_reward'
            for m in agent_spice._model.submodules_sindy
        }
    )
    if idx not in n_params_dict:
        continue
    n_parameters_spice = n_params_dict[idx]

    bic_spice = bayesian_information_criterion(
        data=true_actions,
        probs=probs_spice,
        n_parameters=n_parameters_spice
    )
    aic_spice = 2 * n_parameters_spice - 2 * ll_spice

    metrics_data.append({
        'participant_id': real_pid,
        'nll_spice': -ll_spice,
        'nll_rnn': -ll_rnn,
        'trial_likelihood_spice': spice_per_trial_like,
        'trial_likelihood_rnn': rnn_per_trial_like,
        'bic_spice': bic_spice,
        'aic_spice': aic_spice,
        'n_parameters_spice': n_parameters_spice,
        'metric_n_trials': n_trials_test
    })

metrics_df = pd.DataFrame(metrics_data)
print(f"Number of participants with model metrics: {len(metrics_df)}")

# ─── MERGE 2: KEEP ONLY PARTICIPANTS WHO ALSO HAVE METRICS !!! 
final_df = pd.merge(merged_df, metrics_df, on='participant_id', how='inner')
print(f"After inner‐join with metrics: {len(final_df)} participants")


final_df.to_csv('fixed_sindy_analysis_with_metrics.csv', index=False)
behavior_df.to_csv('behavior_metrics_fixed.csv', index=False)
sindy_df.to_csv('sindy_parameters.csv', index=False)
metrics_df.to_csv('model_evaluation_metrics.csv', index=False)

