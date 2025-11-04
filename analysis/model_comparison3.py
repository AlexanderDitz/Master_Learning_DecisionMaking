#!/usr/bin/env python3
"""
Model Comparison: Predictive Accuracy
Compares LSTM, GQL, Q-Agent, RNN, and SPICE on Dezfouli 2019 behavioral data.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import pandas as pd
from benchmarking.benchmarking_dezfouli2019 import Dezfouli2019GQL, AgentGQL

from utils.model_loading_utils import (
    load_dezfouli_dataset, load_lstm_model, load_gql_model,
    load_rnn_model, load_spice_model
)
from resources.bandits import AgentQ

import inspect

def compute_accuracy(agent, participant_data, debug=False):
    """
    Compute predictive accuracy for a single agent and participant.
    Returns percent correct and negative log-likelihood.
    If debug=True, prints the first 10 predictions and actions.
    """
    actions = participant_data['choice'].values.astype(int)
    rewards = participant_data['reward'].values
    rewards_matrix = np.zeros((len(actions), 2))
    rewards_matrix[np.arange(len(actions)), actions] = rewards

    pred_actions = []
    log_probs = []

    # Determine if agent uses history
    history_agents = {'AgentLSTM', 'AgentGQL', 'AgentNetwork', 'AgentSpice', 'Dezfouli2019GQL', 'RLRNN_dezfouli2019'}
    agent_name = type(agent).__name__
    use_history = agent_name in history_agents or hasattr(agent, 'model_rnn') or hasattr(agent, 'rnn')
    if debug:
        print("DEBUG: agent class name:", agent_name)
        print("DEBUG: use_history:", use_history)

    # Set up session for agents that require it
    if agent_name in ["AgentNetwork", "AgentSpice", "RLRNN_dezfouli2019"]:
        participant_id = getattr(participant_data, 'name', None)
        if participant_id is not None:
            agent.new_sess(participant_id=participant_id)
        else:
            agent.new_sess()

    if agent_name == "AgentGQL" and not hasattr(agent, "_state"):
        agent._model.set_initial_state(batch_size=1)
        state = agent._model.get_state()
        agent._state = {
            'x_value_reward': state['x_value_reward'],
            'x_value_choice': state['x_value_choice'],
        }

    # Trial loop
    for t in range(len(actions)):
        if agent_name == "AgentLSTM":
            pred = agent.get_choice()
            probs = agent.get_choice_probs()
        elif agent_name == "Dezfouli2019GQL":
            pred = agent.get_choice(rewards_matrix[:t+1], actions[:t])
            probs = agent.get_choice_probs(rewards_matrix[:t+1], actions[:t])
        elif agent_name in ["AgentNetwork", "AgentSpice", "RLRNN_dezfouli2019"]:
            pred = agent.get_choice()
            probs = agent.get_choice_probs()
        else:
            pred = agent.get_choice()
            probs = agent.get_choice_probs()

        if hasattr(probs, 'ndim') and probs.ndim > 1:
            probs = probs[-1]
        action_prob = probs[actions[t]] if probs[actions[t]] > 0 else 1e-10
        log_probs.append(np.log(action_prob))
        pred_actions.append(pred)

    pred_actions = np.array(pred_actions)
    nll = -np.sum(log_probs)
    acc = np.mean(pred_actions == actions)

    if debug:
        print("  First 10 true actions:", actions[:10])
        print("  First 10 predicted actions:", pred_actions[:10])
        print("  First 10 log-probs:", log_probs[:10])
        print("  NLL:", nll)

    return acc, nll

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass
    print("🔬 MODEL COMPARISON: Predictive Accuracy")
    print("=" * 80)

    # Load dataset
    print("📂 Loading Dezfouli 2019 dataset...")
    dataset = load_dezfouli_dataset()
    if dataset is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    print(f"✓ Loaded dataset: {len(dataset)} participants")

    # Model paths (update if needed)
    base_params = "params/dezfouli2019"
    lstm_path = os.path.join(base_params, "lstm_dezfouli2019.pkl")
    rnn_path = os.path.join(base_params, "rnn_dezfouli2019_l2_0_001.pkl")
    spice_path = os.path.join(base_params, "spice2_dezfouli2019_l2_0_001.pkl")
    gql_path = os.path.join(base_params, "gql_dezfouli2019_PhiChiBetaKappaC.pkl")

    # Load models
    print("\n🤖 Loading models...")
    models = {}

    def print_model_weights(agent, name):
        try:
            if hasattr(agent, 'model_rnn'):
                weights = agent.model_rnn.state_dict()
            elif hasattr(agent, '_model'):
                weights = agent._model.state_dict()
            elif hasattr(agent, 'model'):
                weights = agent.model.state_dict()
            else:
                print(f"{name}: No accessible model weights.")
                return
            print(f"{name} first few weights:")
            for k, v in list(weights.items())[:3]:
                print(f"  {k}: {v.flatten()[:5].tolist()}")
        except Exception as e:
            print(f"{name}: Could not print weights ({e})")

    agent_lstm = load_lstm_model(lstm_path, deterministic=False)
    if agent_lstm:
        models['LSTM'] = agent_lstm
        print_model_weights(agent_lstm, "LSTM")

    agent_rnn = load_rnn_model(rnn_path, deterministic=False)
    if agent_rnn:
        models['RNN'] = agent_rnn
        print_model_weights(agent_rnn, "RNN")

    agent_spice = load_spice_model(spice_path, rnn_path, deterministic=False)
    if agent_spice:
        models['SPICE'] = agent_spice
        print_model_weights(agent_spice, "SPICE")

    gql_loaded = load_gql_model(gql_path, deterministic=False)
    if gql_loaded:
        agent_gql_list, _ = gql_loaded
        models['GQL'] = agent_gql_list[0]
        print_model_weights(agent_gql_list[0], "GQL")

    models['Q-Agent'] = AgentQ(n_actions=2)

    print(f"\n📊 Models loaded: {list(models.keys())}")

    # Debug: print predictions for the first participant for each model
    debug_participant = list(dataset.keys())[0]
    print(f"\n🔬 Debug: Showing predictions for participant {debug_participant}")
    pdata_debug = dataset[debug_participant]
    for model_name, agent in models.items():
        print(f"Model: {model_name}")
        compute_accuracy(agent, pdata_debug, debug=True)

    import matplotlib.pyplot as plt
    from utils.plotting import plot_session

    # Select only numeric columns (e.g., 'choice', 'reward', etc.)
    numeric_cols = dataset[debug_participant].select_dtypes(include=[np.number]).columns
    experiment_matrix = dataset[debug_participant][numeric_cols].values

    # --- Construct LSTM input: [one-hot choice, one-hot reward] ---
    actions = dataset[debug_participant]['choice'].values.astype(int)
    rewards = dataset[debug_participant]['reward'].values.astype(int)
    n_trials = len(actions)
    lstm_input = np.zeros((n_trials, 4), dtype=np.float32)
    # One-hot for choice (first two columns)
    lstm_input[np.arange(n_trials), actions] = 1
    lstm_input[np.arange(n_trials), actions] = 1
    # One-hot for reward (last two columns)
    lstm_input[np.arange(n_trials), 2 + rewards] = 1

    print("experiment_matrix shape:", experiment_matrix.shape)
    print("experiment_matrix sample:", experiment_matrix[:5])

    # --- Plot traces for the first participant/session with correct input per agent ---
    fig, axs = plot_session(agents={'rnn': models['RNN']}, experiment=experiment_matrix)
    plt.savefig('rnn_trace.png', dpi=300)
    fig, axs = plot_session(agents={'sindy': models['SPICE']}, experiment=experiment_matrix)
    plt.savefig('spice_trace.png', dpi=300)
    fig, axs = plot_session(agents={'lstm': models['LSTM']}, experiment=lstm_input)
    plt.savefig('lstm_trace.png', dpi=300)
    if hasattr(models['GQL'], "new_sess"):
        models['GQL'].new_sess()
    if hasattr(models['GQL'], "_model"):
        models['GQL']._model.set_initial_state(batch_size=1)
    fig, axs = plot_session(agents={'gql': models['GQL']}, experiment=lstm_input)
    plt.savefig('gql_trace.png', dpi=300)
    plt.show()

    # Evaluate predictive accuracy
    print("\n🔎 Evaluating predictive accuracy for each model...")
    results = {model: [] for model in models}

    for pid, pdata in dataset.items():
        for model_name, agent in models.items():
            # 🔁 Reset the agent before each participant session
            if hasattr(agent, "new_sess"):
                agent.new_sess()
            elif hasattr(agent, "reset"):
                agent.reset()

            # 🧠 Special reset for GQL models
            if model_name == "GQL" and hasattr(agent, "_model"):
                agent._model.set_initial_state(batch_size=1)

            # Compute predictive accuracy
            acc, nll = compute_accuracy(agent, pdata)
            results[model_name].append({'participant': pid, 'accuracy': acc, 'nll': nll})

    # Aggregate and print results
    print("\n📈 Predictive Accuracy Results (mean ± std):")
    summary = []
    for model_name, res in results.items():
        accs = [r['accuracy'] for r in res if not np.isnan(r['accuracy'])]
        nlls = [r['nll'] for r in res if not np.isnan(r['nll'])]
        acc_mean, acc_std = np.mean(accs), np.std(accs)
        nll_mean = np.mean(nlls) if nlls else np.nan
        summary.append((model_name, acc_mean, acc_std, nll_mean))
        print(f"{model_name:8s} | Accuracy: {acc_mean:.3f} ± {acc_std:.3f} | NLL: {nll_mean:.2f}")

    # Save summary table
    df_summary = pd.DataFrame(summary, columns=['Model', 'Accuracy_Mean', 'Accuracy_Std', 'NLL_Mean'])
    df_summary.to_csv("model_comparison3_accuracy_summary.csv", index=False)
    print("\n✅ Results saved to model_comparison3_accuracy_summary.csv")

if __name__ == "__main__":
    main()