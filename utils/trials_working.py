import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

n_actions = 2
n_trials = 1000  # Full dataset: 1000 trials per participant
n_participants = 100

from resources.bandits import BanditsDrift, AgentQ, AgentNetwork, create_dataset
from resources.rnn import BaseRNN, RLRNN
from resources.rnn_utils import DatasetRNN
from resources.rnn_training import fit_model
from resources.sindy_training import fit_model as fit_model_sindy

environment = BanditsDrift(sigma=0.2, n_actions=n_actions)

all_xs = []
all_ys = []
participant_agents = []

#  alpha values for each participant (from 0.1 to 0.9)
alpha_values = np.linspace(0.1, 0.9, n_participants)

for participant_id in range(n_participants):
    alpha_reward = alpha_values[participant_id]
    
    agent = AgentQ(
        n_actions=n_actions,
        alpha_reward=alpha_reward,
        alpha_penalty=alpha_reward * 0.8,
        forget_rate=0.2,
    )
    participant_agents.append(agent)
    
    dataset, _, _ = create_dataset(
        agent=agent,
        environment=environment,
        n_trials=n_trials,
        n_sessions=1,
    )
    
    # participant ID in the dataset ( last column)
    dataset.xs[..., -1] = participant_id
    all_xs.append(dataset.xs)
    all_ys.append(dataset.ys)
    
    print(f"Participant {participant_id}: α_reward={alpha_reward:.2f}, α_penalty={alpha_reward*0.8:.2f}")

combined_xs_full = torch.cat(all_xs)
combined_ys_full = torch.cat(all_ys)
combined_dataset_full = DatasetRNN(combined_xs_full, combined_ys_full)

print(f"Combined full dataset shape: {combined_dataset_full.xs.shape}")
print(f"Number of participants: {len(combined_dataset_full.xs[..., -1].unique())}")

# RNN modules and control parameters for SINDy
list_rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
list_control_parameters = ['c_action', 'c_reward', 'c_value_reward']

# SINDy library setup
library_setup = {
    'x_learning_rate_reward': ['c_reward', 'c_value_reward'],
}

# SINDy filter setup
filter_setup = {
    'x_learning_rate_reward': ['c_action', 1, True],
    'x_value_reward_not_chosen': ['c_action', 0, True],
    'x_value_choice_chosen': ['c_action', 1, True],
    'x_value_choice_not_chosen': ['c_action', 0, True],
}

def run_training_and_evaluation(dataset, label):
    print(f"\n==== Running pipeline for dataset: {label} ====")
    
    print("\nTraining RNN...")
    model_rnn = RLRNN(
        n_actions=n_actions, 
        n_participants=n_participants, 
        list_signals=list_rnn_modules + list_control_parameters
    )
    optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=5e-3)
    
    model_rnn, _, _ = fit_model(
        model=model_rnn,
        optimizer=optimizer_rnn,
        dataset_train=dataset,
        epochs=4096,
        n_steps=16,
        scheduler=True,
        convergence_threshold=0,
    )
    agent_rnn = AgentNetwork(model_rnn=model_rnn, n_actions=n_actions)
    
    print("\nFitting SINDy...")
    agent_sindy = fit_model_sindy(
        rnn_modules=list_rnn_modules,
        control_parameters=list_control_parameters,
        agent=agent_rnn,
        data=environment,
        library_setup=library_setup,
        filter_setup=filter_setup,
        verbose=True,
    )
    
    print("\nEvaluating models for each participant...")
    
    # Helper function to compute BIC for a single participant.
    def compute_bic(agent, xs, ys):
        #model (in _model or model_rnn)
        model = agent._model if hasattr(agent, '_model') else agent.model_rnn
        model.eval()
        with torch.no_grad():
            # If xs is 2D, add a batch dimension to match expected input shape (batch, trials, features)
            if xs.dim() == 2:
                xs = xs.unsqueeze(0)
            logits, _ = model(xs)
            # Squeeze the batch dimension: resulting shape should be (n_trials, n_actions)
            logits = logits.squeeze(0)
        # logits to probabilities via softmax
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        #  observed choices to a NumPy array and flattento get scalar values
        choices = ys.cpu().numpy().astype(int).flatten()
        # choices length matches the number of predictions (probs)
        if len(choices) > probs.shape[0]:
            choices = choices[:probs.shape[0]]
        # For each trial, get the probability of the chosen action
        chosen_probs = np.array([probs[i, int(choice)] for i, choice in enumerate(choices)])
        # lg-likelihood 
        ll = np.sum(np.log(chosen_probs + 1e-8))
        # Count number of trainable parameters 
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Number of data points (per participant)
        n_data = len(choices)
        # BIC = -2 * ll + n_params * log(n_data)
        bic = -2 * ll + n_params * np.log(n_data)
        return bic
    
    # Loop over participants in the current dataset subset
    bic_values = []
    for participant_id in range(n_participants):
        print(f"Evaluating participant {participant_id}...")
        # Filter the dataset for the current participant ( in the last column)
        mask = (dataset.xs[..., -1] == participant_id)
        xs_participant = dataset.xs[mask]
        ys_participant = dataset.ys[mask]
        
        bic = compute_bic(agent_rnn, xs_participant, ys_participant)
        print(f"Participant {participant_id} BIC: {bic:.2f}")
        bic_values.append(bic)
    
    avg_bic = np.mean(bic_values)
    print(f"\nAverage BIC for {label}: {avg_bic:.2f}")
    
    print("\nIdentified SINDy equations:")
    for module_name in list_rnn_modules:
        print(f"Module: {module_name}")
        if hasattr(agent_sindy, f'sindy_{module_name}'):
            sindy_model = getattr(agent_sindy, f'sindy_{module_name}')
            print(f"  Equation: {sindy_model}")
        else:
            print("  Module not found in SINDy agent")
    
    return avg_bic

#  storing the results (subset of trials + corresponding avge BIC)
bic_results = []

#  pipeline for increasing trial subsets from 100 to 1000 in steps of 100.
for subset_length in range(100, n_trials + 1, 100):
    subset_xs = []
    subset_ys = []
    for participant_id in range(n_participants):
        start_index = participant_id * n_trials
        subset_xs.append(combined_xs_full[start_index:start_index + subset_length])
        subset_ys.append(combined_ys_full[start_index:start_index + subset_length])
    
    combined_xs_subset = torch.cat(subset_xs)
    combined_ys_subset = torch.cat(subset_ys)
    combined_dataset_subset = DatasetRNN(combined_xs_subset, combined_ys_subset)
    
    label = f"{subset_length} trials per participant"
    print(f"\n{'='*40}\nStarting run: {label}")
    avg_bic = run_training_and_evaluation(combined_dataset_subset, label)
    
    bic_results.append((subset_length, avg_bic))

trial_lengths, avg_bic_values = zip(*bic_results)
plt.figure(figsize=(8, 5))
plt.plot(trial_lengths, avg_bic_values, marker='o')
plt.xlabel('Trials per Participant')
plt.ylabel('Average BIC')
plt.title('Average BIC vs. Trials per Participant')
plt.grid(True)
plt.show()
