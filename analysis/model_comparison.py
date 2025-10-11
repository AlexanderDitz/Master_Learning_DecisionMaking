"""
Model Comparison: LSTM vs SPICE with L2=0.001

This script compares the performance of LSTM and SPICE models on the dezfouli2019 dataset.
"""

import sys
import os
import torch
import pickle
import numpy as np
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from benchmarking.benchmarking_lstm import RLLSTM, AgentLSTM, setup_agent_lstm
from utils.setup_agents import setup_agent_spice
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentSpice
from resources.sindy_utils import load_spice
from resources.rnn import RLRNN_eckstein2022


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


def main():
    """
    Main function to compare LSTM and SPICE models.
    """
    # Define model paths
    base_path = "params/dezfouli2019"
    lstm_path = os.path.join(base_path, "lstm_dezfouli2019.pkl")
    spice_path = os.path.join(base_path, "spice_dezfouli2019_l2_0_001.pkl")
    rnn_path = os.path.join(base_path, "rnn_dezfouli2019_l2_0_001.pkl")  # RNN model for SPICE
    
    print("=== Model Comparison: LSTM vs SPICE ===")
    print()
    
    # Load LSTM model
    try:
        lstm_agent = load_lstm_model(lstm_path)
        print(f"✓ LSTM model loaded successfully")
        print(f"  - Model type: {type(lstm_agent._model)}")
        print(f"  - Number of actions: {lstm_agent._n_actions}")
        print()
    except Exception as e:
        print(f"✗ Failed to load LSTM model: {e}")
        lstm_agent = None
    
    # Load SPICE model
    try:
        spice_agent = load_spice_model(spice_path, rnn_path)
        print(f"✓ SPICE model loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load SPICE model: {e}")
        print(f"Error details: {str(e)}")
        spice_agent = None
    
    # Check if both models loaded successfully
    if lstm_agent is not None and spice_agent is not None:
        print("🎉 Both models loaded successfully!")
        print("Ready for comparison...")
        
        # TODO: Implement model comparison
        print("\nModel comparison metrics will be implemented next...")
    else:
        print("❌ Could not load both models. Comparison cannot proceed.")
        if lstm_agent is None:
            print("  - LSTM model failed to load")
        if spice_agent is None:
            print("  - SPICE model failed to load")


if __name__ == "__main__":
    main()
