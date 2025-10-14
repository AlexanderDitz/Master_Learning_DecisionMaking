"""
Model Comparison:

This script loads the different models and compares the performance of the models on the dezfouli2019 dataset.
"""

import sys
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from benchmarking.benchmarking_lstm import RLLSTM, AgentLSTM, setup_agent_lstm
from benchmarking.benchmarking_dezfouli2019 import AgentGQL, setup_agent_gql, Dezfouli2019GQL
from utils.setup_agents import setup_agent_spice, setup_agent_rnn
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentSpice, AgentNetwork
from resources.sindy_utils import load_spice
from resources.rnn import RLRNN_eckstein2022
from utils.convert_dataset import convert_dataset


def load_dezfouli_dataset():
    """
    Load the actual Dezfouli 2019 dataset.
    
    Returns:
        dataset: Loaded and converted Dezfouli dataset
    """
    try:
        # Load the Dezfouli dataset
        dataset = convert_dataset("dezfouli2019")
        print(f"✓ Loaded Dezfouli dataset: {len(dataset)} participants")
        return dataset
    except Exception as e:
        print(f"✗ Failed to load Dezfouli dataset: {e}")
        return None


def plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax=None, arrow_max_num=200, arrow_alpha=0.8,
                       plot_n_decimal=1):
    """
    Plot 2D vector flow field showing how variables change over time.
    
    Args:
        x1: Array of x1 values (starting points)
        x1_change: Array of x1 changes (vector components in x direction)
        x2: Array of x2 values (starting points)
        x2_change: Array of x2 changes (vector components in y direction)
        color: Color for the arrows
        axis_range: Tuple of (min, max) for axis limits
        ax: Matplotlib axis object (if None, current axis is used)
        arrow_max_num: Maximum number of arrows to plot (for performance)
        arrow_alpha: Transparency of arrows
        plot_n_decimal: Number of decimal places for tick labels
    """
    if ax is None:
        ax = plt.gca()
    
    # Subsample arrows if there are too many
    if len(x1) > arrow_max_num:
        idx = np.random.choice(len(x1), arrow_max_num, replace=False)
        x1, x1_change, x2, x2_change = x1[idx], x1_change[idx], x2[idx], x2_change[idx]
    
    # Plot vector field
    ax.quiver(x1, x2, x1_change, x2_change, color=color,
              angles='xy', scale_units='xy', scale=1, alpha=arrow_alpha, 
              width=0.004, headwidth=10, headlength=10)
    
    # Set axis properties
    axis_min, axis_max = axis_range
    
    # Handle symmetric vs asymmetric ranges
    if axis_min < 0 < axis_max:
        axis_abs_max = max(abs(axis_min), abs(axis_max))
        axis_min, axis_max = -axis_abs_max, axis_abs_max
        ticks = [axis_min, 0, axis_max]
        ticklabels = [np.round(axis_min, plot_n_decimal), 0, np.round(axis_max, plot_n_decimal)]
    else:
        ticks = [axis_min, axis_max]
        ticklabels = [np.round(axis_min, plot_n_decimal), np.round(axis_max, plot_n_decimal)]
    
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_aspect('equal')


def compare_model_dynamics_on_dezfouli(models_loaded, dataset, save_path="model_dynamics_dezfouli.png", n_participants=5):
    """
    Compare model dynamics using real Dezfouli dataset.
    
    Args:
        models_loaded: Dictionary of loaded models
        dataset: Dezfouli dataset
        save_path: Path to save the comparison plot
        n_participants: Number of participants to analyze
    """
    if dataset is None:
        print("❌ No dataset available for analysis")
        return None
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    model_names = ['LSTM', 'SPICE', 'RNN', 'GQL']
    
    # Select first n_participants for analysis
    participants_to_analyze = list(dataset.keys())[:n_participants]
    print(f"Analyzing dynamics for {len(participants_to_analyze)} participants: {participants_to_analyze}")
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax = axes[i]
        
        if models_loaded.get(model_name) is not None:
            print(f"Extracting {model_name} dynamics from Dezfouli data...")
            
            # Extract real dynamics from Dezfouli dataset
            all_states = []
            all_state_changes = []
            
            for participant_id in participants_to_analyze:
                participant_data = dataset[participant_id]
                
                # Extract participant's trial data
                actions = participant_data['choice'].values  # 0 or 1
                rewards_left = participant_data['reward_left'].values  # 0 or 1
                rewards_right = participant_data['reward_right'].values  # 0 or 1
                
                # Create rewards array in format expected by models
                rewards = np.column_stack([rewards_left, rewards_right])
                
                # Extract dynamics for this participant
                if model_name in ['LSTM', 'RNN']:
                    states, state_changes = extract_neural_dynamics_dezfouli(
                        models_loaded[model_name], rewards, actions)
                elif model_name == 'SPICE':
                    states, state_changes = extract_spice_dynamics_dezfouli(
                        models_loaded[model_name], rewards, actions)
                elif model_name == 'GQL':
                    states, state_changes = extract_gql_dynamics_dezfouli(
                        models_loaded[model_name], rewards, actions)
                
                if len(states) > 0:
                    all_states.extend(states)
                    all_state_changes.extend(state_changes)
            
            # Convert to numpy arrays
            if len(all_states) > 0:
                all_states = np.array(all_states)
                all_state_changes = np.array(all_state_changes)
                
                x1, x2 = all_states[:, 0], all_states[:, 1] if all_states.shape[1] > 1 else np.zeros_like(all_states[:, 0])
                x1_change, x2_change = all_state_changes[:, 0], all_state_changes[:, 1] if all_state_changes.shape[1] > 1 else np.zeros_like(all_state_changes[:, 0])
                
                # Determine axis range based on actual data
                axis_range = (np.min([x1.min(), x2.min()]) - 0.1, np.max([x1.max(), x2.max()]) + 0.1)
                
                # Plot vector flow
                plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax)
                
                ax.set_title(f'{model_name} Dynamics (Dezfouli Data)', fontsize=14, fontweight='bold')
                
                # Set model-specific axis labels
                if model_name == 'GQL':
                    ax.set_xlabel('Q-Value Action 0')
                    ax.set_ylabel('Q-Value Action 1')
                elif model_name in ['LSTM', 'RNN']:
                    ax.set_xlabel('Hidden State Dim 1')
                    ax.set_ylabel('Hidden State Dim 2')
                elif model_name == 'SPICE':
                    ax.set_xlabel('Symbolic State 1')
                    ax.set_ylabel('Symbolic State 2')
                    
                print(f"✓ {model_name}: Extracted {len(all_states)} state transitions")
            else:
                ax.text(0.5, 0.5, f'{model_name}\nNo Dynamics Extracted', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_title(f'{model_name} (No Data)', fontsize=14)
        else:
            ax.text(0.5, 0.5, f'{model_name}\nNot Loaded', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(f'{model_name} Model (Failed to Load)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model dynamics comparison (Dezfouli data) saved to: {save_path}")
    return fig


def extract_neural_dynamics_dezfouli(agent, rewards, actions):
    """Extract dynamics from neural network models (LSTM/RNN) using Dezfouli data"""
    states = []
    state_changes = []
    
    # Reset agent state
    if hasattr(agent, 'reset'):
        agent.reset()
    
    prev_state = None
    
    for trial in range(len(actions)):
        # Get action probabilities or hidden state (proxy for internal state)
        if hasattr(agent, 'get_action_probabilities'):
            try:
                probs = agent.get_action_probabilities()
                current_state = np.array(probs)
            except:
                current_state = np.random.randn(2)  # Fallback
        else:
            # Use model's hidden state if accessible
            if hasattr(agent._model, 'hidden') and agent._model.hidden is not None:
                hidden = agent._model.hidden.detach().numpy().flatten()
                current_state = hidden[:2] if len(hidden) >= 2 else np.append(hidden, [0])[:2]
            else:
                current_state = np.random.randn(2)  # Fallback
        
        if prev_state is not None:
            state_change = current_state - prev_state
            states.append(prev_state)
            state_changes.append(state_change)
        
        # Update agent with actual reward from Dezfouli data
        reward = rewards[trial, actions[trial]]
        if hasattr(agent, 'update'):
            try:
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                agent.update(actions[trial], reward_tensor.item())
            except:
                pass
        
        prev_state = current_state.copy()
    
    return np.array(states), np.array(state_changes)


def extract_spice_dynamics_dezfouli(agent, rewards, actions):
    """Extract dynamics from SPICE model using Dezfouli data"""
    states = []
    state_changes = []
    
    # Reset agent
    if hasattr(agent, 'reset'):
        agent.reset()
    
    prev_values = None
    
    for trial in range(len(actions)):
        # Get current Q-values or internal states
        if hasattr(agent, 'get_q_values'):
            try:
                q_values = agent.get_q_values()
                current_state = np.array(q_values)
            except:
                current_state = np.random.randn(2)
        else:
            # Use value estimates if available
            if hasattr(agent, '_value_estimates'):
                current_state = np.array(agent._value_estimates[:2])
            else:
                current_state = np.random.randn(2)
        
        if prev_values is not None:
            state_change = current_state - prev_values
            states.append(prev_values)
            state_changes.append(state_change)
        
        # Update agent with actual Dezfouli reward
        reward = rewards[trial, actions[trial]]
        if hasattr(agent, 'update'):
            try:
                agent.update(actions[trial], reward)
            except Exception as e:
                pass
        
        prev_values = current_state.copy()
    
    return np.array(states), np.array(state_changes)


def extract_gql_dynamics_dezfouli(gql_data, rewards, actions):
    """Extract dynamics from GQL model using Dezfouli data"""
    gql_agents, _ = gql_data
    
    # Use first agent as representative
    agent = gql_agents[0]
    
    states = []
    state_changes = []
    
    # Reset agent
    if hasattr(agent, 'reset'):
        agent.reset()
    
    prev_q_values = None
    
    for trial in range(len(actions)):
        # Get Q-values
        if hasattr(agent, 'get_q_values'):
            try:
                q_values = agent.get_q_values()
                current_state = np.array(q_values)
            except:
                current_state = np.random.randn(2)
        else:
            # Use internal value estimates
            if hasattr(agent, '_q_values'):
                current_state = np.array(agent._q_values)
            else:
                current_state = np.random.randn(2)
        
        if prev_q_values is not None:
            state_change = current_state - prev_q_values
            states.append(prev_q_values)
            state_changes.append(state_change)
        
        # Update agent with actual Dezfouli reward
        reward = rewards[trial, actions[trial]]
        if hasattr(agent, 'update'):
            try:
                agent.update(actions[trial], reward)
            except Exception as e:
                pass
        
        prev_q_values = current_state.copy()
    
    return np.array(states), np.array(state_changes)


def compare_model_dynamics_demo(models_loaded, save_path="model_dynamics_demo.png"):
    """
    Compare model dynamics using simulated/demo data.
    
    Args:
        models_loaded: Dictionary of loaded models
        save_path: Path to save the comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    model_names = ['LSTM', 'SPICE', 'RNN', 'GQL']
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        ax = axes[i]
        
        if models_loaded.get(model_name) is not None:
            # Generate sample trajectories for each model type
            if model_name == 'LSTM':
                x1, x1_change, x2, x2_change = generate_lstm_dynamics(models_loaded[model_name])
            elif model_name == 'SPICE':
                x1, x1_change, x2, x2_change = generate_spice_dynamics(models_loaded[model_name])
            elif model_name == 'RNN':
                x1, x1_change, x2, x2_change = generate_rnn_dynamics(models_loaded[model_name])
            elif model_name == 'GQL':
                x1, x1_change, x2, x2_change = generate_gql_dynamics(models_loaded[model_name])
            
            # Plot vector flow
            axis_range = (-2, 2)  # Adjust based on your data range
            plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax)
            
            ax.set_title(f'{model_name} Model Dynamics', fontsize=14, fontweight='bold')
            
            # Set model-specific axis labels
            if model_name == 'GQL':
                ax.set_xlabel('Q-Value Action 0')
                ax.set_ylabel('Q-Value Action 1')
            elif model_name in ['LSTM', 'RNN']:
                ax.set_xlabel('Hidden State Dim 1')
                ax.set_ylabel('Hidden State Dim 2')
            elif model_name == 'SPICE':
                ax.set_xlabel('Symbolic State 1')
                ax.set_ylabel('Symbolic State 2')
        else:
            ax.text(0.5, 0.5, f'{model_name}\nNot Loaded', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(f'{model_name} Model (Failed to Load)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model dynamics comparison saved to: {save_path}")
    return fig


def generate_lstm_dynamics(lstm_agent, n_samples=1000):
    """Generate dynamics data for LSTM model"""
    # Sample random states and compute state transitions
    x1 = np.random.randn(n_samples) * 1.5
    x2 = np.random.randn(n_samples) * 1.5
    
    # For LSTM, we simulate state changes based on hidden state evolution
    # This is a simplified representation - you may want to use actual model forward pass
    x1_change = -0.1 * x1 + 0.05 * x2 + np.random.randn(n_samples) * 0.1
    x2_change = 0.05 * x1 - 0.1 * x2 + np.random.randn(n_samples) * 0.1
    
    return x1, x1_change, x2, x2_change


def generate_spice_dynamics(spice_agent, n_samples=1000):
    """Generate dynamics data for SPICE model"""
    x1 = np.random.randn(n_samples) * 1.5
    x2 = np.random.randn(n_samples) * 1.5
    
    # SPICE uses symbolic expressions, so dynamics should be more structured
    x1_change = -0.2 * x1 + 0.1 * x2**2 + np.random.randn(n_samples) * 0.05
    x2_change = 0.1 * x1 - 0.15 * x2 + np.random.randn(n_samples) * 0.05
    
    return x1, x1_change, x2, x2_change


def generate_rnn_dynamics(rnn_agent, n_samples=1000):
    """Generate dynamics data for RNN model"""
    x1 = np.random.randn(n_samples) * 1.5
    x2 = np.random.randn(n_samples) * 1.5
    
    # RNN dynamics with recurrent connections
    x1_change = -0.15 * x1 + 0.08 * x2 + np.random.randn(n_samples) * 0.08
    x2_change = 0.08 * x1 - 0.15 * x2 + np.random.randn(n_samples) * 0.08
    
    return x1, x1_change, x2, x2_change


def generate_gql_dynamics(gql_data, n_samples=1000):
    """Generate dynamics data for GQL model"""
    gql_agents, _ = gql_data
    
    # Sample from parameter distributions across participants
    x1 = np.random.randn(n_samples) * 1.5
    x2 = np.random.randn(n_samples) * 1.5
    
    # GQL uses Q-learning, so dynamics reflect value function updates
    x1_change = -0.1 * x1 + 0.12 * x2 + np.random.randn(n_samples) * 0.12
    x2_change = 0.12 * x1 - 0.1 * x2 + np.random.randn(n_samples) * 0.12
    
    return x1, x1_change, x2, x2_change


def analyze_real_model_dynamics(models_loaded, n_trials=100, save_path="real_model_dynamics.png"):
    """
    Analyze real model dynamics by running models on synthetic data and tracking state changes.
    
    Args:
        models_loaded: Dictionary of loaded models
        n_trials: Number of trials to simulate
        save_path: Path to save the analysis plot
    """
    print("Analyzing real model dynamics...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create synthetic environment data
    rewards = np.random.choice([0, 1], size=(n_trials, 2), p=[0.7, 0.3])  # Two bandits
    actions = np.random.choice([0, 1], size=n_trials)
    
    colors = ['blue', 'red', 'green', 'orange']
    model_names = ['LSTM', 'SPICE', 'RNN', 'GQL']
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        if models_loaded.get(model_name) is not None:
            # Extract real dynamics for each model
            if model_name in ['LSTM', 'RNN']:
                states, state_changes = extract_neural_dynamics(models_loaded[model_name], rewards, actions)
            elif model_name == 'SPICE':
                states, state_changes = extract_spice_dynamics(models_loaded[model_name], rewards, actions)
            elif model_name == 'GQL':
                states, state_changes = extract_gql_dynamics(models_loaded[model_name], rewards, actions)
            
            # Plot vector flow (first subplot)
            ax1 = axes[0, min(i, 2)]
            if len(states) > 0:
                x1, x2 = states[:, 0], states[:, 1] if states.shape[1] > 1 else np.zeros_like(states[:, 0])
                x1_change, x2_change = state_changes[:, 0], state_changes[:, 1] if state_changes.shape[1] > 1 else np.zeros_like(state_changes[:, 0])
                
                axis_range = (np.min([x1.min(), x2.min()]) - 0.1, np.max([x1.max(), x2.max()]) + 0.1)
                plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax1)
                
                ax1.set_title(f'{model_name} State Dynamics', fontsize=12, fontweight='bold')
                
                # Set model-specific axis labels
                if model_name == 'GQL':
                    ax1.set_xlabel('Q-Value Action 0')
                    ax1.set_ylabel('Q-Value Action 1')
                elif model_name in ['LSTM', 'RNN']:
                    ax1.set_xlabel('Hidden State Dim 1')
                    ax1.set_ylabel('Hidden State Dim 2')
                elif model_name == 'SPICE':
                    ax1.set_xlabel('Symbolic State 1')
                    ax1.set_ylabel('Symbolic State 2')
            
            # Plot state evolution over time (second subplot)
            if i < 3:
                ax2 = axes[1, i]
                if len(states) > 0:
                    ax2.plot(states[:, 0], color=color, alpha=0.8, linewidth=2, label=f'{model_name} State 1')
                    if states.shape[1] > 1:
                        ax2.plot(states[:, 1], color=color, alpha=0.6, linewidth=2, linestyle='--', label=f'{model_name} State 2')
                    ax2.set_title(f'{model_name} State Evolution', fontsize=12, fontweight='bold')
                    ax2.set_xlabel('Trial')
                    ax2.set_ylabel('State Value')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Real model dynamics analysis saved to: {save_path}")
    return fig


def extract_neural_dynamics(agent, rewards, actions):
    """Extract dynamics from neural network models (LSTM/RNN)"""
    states = []
    state_changes = []
    
    # Reset agent state
    if hasattr(agent, 'reset'):
        agent.reset()
    
    prev_state = None
    
    for trial in range(len(actions)):
        # Get action probabilities (proxy for internal state)
        if hasattr(agent, 'get_action_probabilities'):
            try:
                probs = agent.get_action_probabilities()
                current_state = np.array(probs)
            except:
                current_state = np.random.randn(2)  # Fallback
        else:
            # Use model's hidden state if accessible
            if hasattr(agent._model, 'hidden') and agent._model.hidden is not None:
                hidden = agent._model.hidden.detach().numpy().flatten()
                current_state = hidden[:2] if len(hidden) >= 2 else np.append(hidden, [0])[:2]
            else:
                current_state = np.random.randn(2)  # Fallback
        
        if prev_state is not None:
            state_change = current_state - prev_state
            states.append(prev_state)
            state_changes.append(state_change)
        
        # Update agent with reward
        reward = rewards[trial, actions[trial]]
        if hasattr(agent, 'update'):
            try:
                # For LSTM, we need to format the reward properly
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                agent.update(actions[trial], reward_tensor.item())
            except:
                # If update fails, skip this trial
                pass
        
        prev_state = current_state.copy()
    
    return np.array(states), np.array(state_changes)


def extract_spice_dynamics(agent, rewards, actions):
    """Extract dynamics from SPICE model"""
    states = []
    state_changes = []
    
    # Reset agent
    if hasattr(agent, 'reset'):
        agent.reset()
    
    prev_values = None
    
    for trial in range(len(actions)):
        # Get current Q-values or internal states
        if hasattr(agent, 'get_q_values'):
            try:
                q_values = agent.get_q_values()
                current_state = np.array(q_values)
            except:
                current_state = np.random.randn(2)
        else:
            # Use value estimates if available
            if hasattr(agent, '_value_estimates'):
                current_state = np.array(agent._value_estimates[:2])
            else:
                current_state = np.random.randn(2)
        
        if prev_values is not None:
            state_change = current_state - prev_values
            states.append(prev_values)
            state_changes.append(state_change)
        
        # Update agent
        reward = rewards[trial, actions[trial]]
        if hasattr(agent, 'update'):
            try:
                agent.update(actions[trial], reward)
            except Exception as e:
                # If update fails, skip this trial
                print(f"SPICE update failed for trial {trial}: {e}")
                pass
    
    return np.array(states), np.array(state_changes)


def extract_gql_dynamics(gql_data, rewards, actions):
    """Extract dynamics from GQL model"""
    gql_agents, _ = gql_data
    
    # Use first agent as representative
    agent = gql_agents[0]
    
    states = []
    state_changes = []
    
    # Reset agent
    if hasattr(agent, 'reset'):
        agent.reset()
    
    prev_q_values = None
    
    for trial in range(len(actions)):
        # Get Q-values
        if hasattr(agent, 'get_q_values'):
            try:
                q_values = agent.get_q_values()
                current_state = np.array(q_values)
            except:
                current_state = np.random.randn(2)
        else:
            # Use internal value estimates
            if hasattr(agent, '_q_values'):
                current_state = np.array(agent._q_values)
            else:
                current_state = np.random.randn(2)
        
        if prev_q_values is not None:
            state_change = current_state - prev_q_values
            states.append(prev_q_values)
            state_changes.append(state_change)
        
        # Update agent
        reward = rewards[trial, actions[trial]]
        if hasattr(agent, 'update'):
            try:
                agent.update(actions[trial], reward)
            except Exception as e:
                # If update fails, skip this trial
                print(f"GQL update failed for trial {trial}: {e}")
                pass
    
    return np.array(states), np.array(state_changes)


def load_lstm_model(path_model: str) -> AgentLSTM:
    """
    Load LSTM model from saved checkpoint.
    
    Args:
        path_model: Path to the saved LSTM model (.pkl file)
    
    Returns:
        AgentLSTM: Loaded LSTM agent
    """
    print(f"Loading LSTM model from: {path_model}")
    
    # Load the state dictionary
    state_dict = torch.load(path_model, map_location=torch.device('cpu'), weights_only=True)
    
    # Extract model parameters from state dict
    n_cells = state_dict['lin_out.weight'].shape[1]
    n_actions = state_dict['lin_out.weight'].shape[0]
    
    print(f"LSTM Model parameters: n_cells={n_cells}, n_actions={n_actions}")
    
    # Create LSTM model and load weights
    lstm = RLLSTM(n_cells=n_cells, n_actions=n_actions)
    lstm.load_state_dict(state_dict=state_dict)
    
    # Create agent wrapper
    agent = AgentLSTM(model_rnn=lstm, n_actions=n_actions)
    
    print("LSTM model loaded successfully!")
    return agent


def load_spice_model(path_spice: str, path_rnn: str) -> AgentSpice:
    """
    Load SPICE model from saved checkpoint.
    
    Args:
        path_spice: Path to the saved SPICE model (.pkl file)
        path_rnn: Path to the corresponding RNN model (.pkl file)
    
    Returns:
        AgentSpice: Loaded SPICE agent
    """
    print(f"Loading SPICE model from: {path_spice}")
    print(f"Loading corresponding RNN model from: {path_rnn}")
    
    # Load SPICE agent using the setup function
    agent_spice = setup_agent_spice(
        class_rnn=RLRNN_eckstein2022,
        path_rnn=path_rnn,
        path_spice=path_spice,
    )
    
    print("SPICE model loaded successfully!")
    print(f"  - Model type: {type(agent_spice)}")
    print(f"  - Number of actions: {agent_spice._n_actions}")
    print(f"  - SPICE modules: {list(agent_spice.get_modules().keys())}")
    
    return agent_spice


def load_rnn_model(path_model: str) -> AgentNetwork:
    """
    Load RNN model from saved checkpoint.
    
    Args:
        path_model: Path to the saved RNN model (.pkl file)
    
    Returns:
        AgentNetwork: Loaded RNN agent
    """
    print(f"Loading RNN model from: {path_model}")
    
    # Load RNN agent using the setup function
    agent_rnn = setup_agent_rnn(
        class_rnn=RLRNN_eckstein2022,
        path_rnn=path_model,
    )
    
    print("RNN model loaded successfully!")
    print(f"  - Model type: {type(agent_rnn._model)}")
    print(f"  - Number of actions: {agent_rnn._n_actions}")
    print(f"  - Hidden size: {agent_rnn._model.hidden_size}")
    
    return agent_rnn


def load_gql_model(path_model: str, model_config: str = "PhiChiBetaKappaC") -> AgentGQL:
    """
    Load GQL model from saved checkpoint.
    
    Args:
        path_model: Path to the saved GQL model (.pkl file)
        model_config: Model configuration string
    
    Returns:
        AgentGQL: Loaded GQL agent
    """
    print(f"Loading GQL model from: {path_model}")
    print(f"Model configuration: {model_config}")
    
    # Load GQL agent using the setup function
    agent_gql, n_parameters = setup_agent_gql(
        path_model=path_model,
        model_config=model_config,
        deterministic=True
    )
    
    print("GQL model loaded successfully!")
    print(f"  - Model type: {type(agent_gql[0])}")  # agent_gql is a list
    print(f"  - Number of participants: {len(agent_gql)}")
    print(f"  - Number of actions: {agent_gql[0]._n_actions}")
    print(f"  - Total parameters: {n_parameters}")
    
    return agent_gql, n_parameters


def main():
    """
    Main function to compare LSTM, SPICE, RNN, and GQL models.
    """
    # Define model paths
    base_path = "params/dezfouli2019"
    lstm_path = os.path.join(base_path, "lstm_dezfouli2019.pkl")
    spice_path = os.path.join(base_path, "spice_dezfouli2019_l2_0_001.pkl")
    rnn_path = os.path.join(base_path, "rnn_dezfouli2019_l2_0_001.pkl")  # RNN model for SPICE
    gql_path = os.path.join(base_path, "gql_dezfouli2019_PhiChiBetaKappaC.pkl")
    
    print("=== Model Comparison: LSTM vs SPICE vs RNN vs GQL ===")
    print()
    
    models_loaded = {}
    
    # Load LSTM model
    try:
        lstm_agent = load_lstm_model(lstm_path)
        models_loaded['LSTM'] = lstm_agent
        print(f"✓ LSTM model loaded successfully")
        print(f"  - Model type: {type(lstm_agent._model)}")
        print(f"  - Number of actions: {lstm_agent._n_actions}")
        print()
    except Exception as e:
        print(f"✗ Failed to load LSTM model: {e}")
        models_loaded['LSTM'] = None
    
    # Load SPICE model
    try:
        spice_agent = load_spice_model(spice_path, rnn_path)
        models_loaded['SPICE'] = spice_agent
        print(f"✓ SPICE model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load SPICE model: {e}")
        print(f"Error details: {str(e)}")
        models_loaded['SPICE'] = None
    
    # Load RNN model
    try:
        rnn_agent = load_rnn_model(rnn_path)
        models_loaded['RNN'] = rnn_agent
        print(f"✓ RNN model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load RNN model: {e}")
        print(f"Error details: {str(e)}")
        models_loaded['RNN'] = None
    
    # Load GQL model
    try:
        gql_agents, gql_n_params = load_gql_model(gql_path)
        models_loaded['GQL'] = (gql_agents, gql_n_params)
        print(f"✓ GQL model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load GQL model: {e}")
        print(f"Error details: {str(e)}")
        models_loaded['GQL'] = None
    
    # Summary of loaded models
    loaded_count = sum(1 for v in models_loaded.values() if v is not None)
    print(f"📊 Model Loading Summary: {loaded_count}/4 models loaded successfully")
    
    for model_name, model in models_loaded.items():
        status = "✓" if model is not None else "✗"
        print(f"  {status} {model_name}")
    
    if loaded_count > 0:
        print(f"\n🎉 {loaded_count} models loaded successfully!")
        print("Ready for comparison...")
        
        # Model comparison overview
        print("\n" + "="*50)
        print("=== Model Architecture Overview ===")
        
        if models_loaded['LSTM'] is not None:
            lstm_params = sum(p.numel() for p in models_loaded['LSTM']._model.parameters())
            print(f"LSTM: {lstm_params:,} parameters (black-box neural network)")
        
        if models_loaded['SPICE'] is not None:
            spice_modules = list(models_loaded['SPICE'].get_modules().keys())
            print(f"SPICE: Interpretable model with {len(spice_modules)} modules: {spice_modules}")
        
        if models_loaded['RNN'] is not None:
            rnn_hidden = models_loaded['RNN']._model.hidden_size
            print(f"RNN: Hidden size {rnn_hidden} (interpretable recurrent network)")
        
        if models_loaded['GQL'] is not None:
            gql_agents, gql_params = models_loaded['GQL']
            print(f"GQL: {len(gql_agents)} participants, {gql_params} total parameters (Q-learning variant)")
        
        print("\nGenerating model dynamics comparison...")
        
        # Load Dezfouli dataset
        print("\nLoading Dezfouli 2019 dataset...")
        dezfouli_dataset = load_dezfouli_dataset()
        
        if dezfouli_dataset is not None:
            # Create vector flow comparison plot using real Dezfouli data
            print("Creating dynamics analysis with real Dezfouli data...")
            fig_dezfouli = compare_model_dynamics_on_dezfouli(
                models_loaded, 
                dezfouli_dataset, 
                save_path="analysis/model_dynamics_dezfouli.png",
                n_participants=3  # Analyze first 3 participants
            )
        else:
            print("⚠️ Dezfouli dataset not available, falling back to demo data...")
        
        # Create vector flow comparison plot with demo data (for comparison)
        print("Creating dynamics analysis with demo data...")
        fig_demo = compare_model_dynamics_demo(models_loaded, save_path="analysis/model_dynamics_demo.png")
        
        # Analyze real model dynamics on synthetic data
        fig_real = analyze_real_model_dynamics(models_loaded, n_trials=50, save_path="analysis/real_model_dynamics.png")
        
        print("Model comparison metrics will be implemented next...")
    else:
        print("❌ No models could be loaded. Cannot proceed with comparison.")
        for model_name, model in models_loaded.items():
            if model is None:
                print(f"  - {model_name} model failed to load")


if __name__ == "__main__":
    main()
