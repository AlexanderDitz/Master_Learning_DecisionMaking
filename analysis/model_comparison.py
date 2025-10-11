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


def load_spice_model(path_model: str, path_rnn: str = None) -> AgentSpice:
    """
    Load SPICE model from saved checkpoint.
    
    Args:
        path_model: Path to the saved SPICE model (.pkl file)
        path_rnn: Path to the RNN model (if needed for SPICE setup)
    
    Returns:
        AgentSpice: Loaded SPICE agent
    """
    print(f"Loading SPICE model from: {path_model}")
    
    # Load SPICE modules
    spice_modules = load_spice(path_model)
    
    # For now, we'll implement the basic loading
    # This might need adjustment based on the actual SPICE model structure
    print("SPICE model loaded successfully!")
    
    # TODO: Complete SPICE loading implementation
    # This will be implemented in the next step
    return None


def main():
    """
    Main function to compare LSTM and SPICE models.
    """
    # Define model paths
    base_path = "params/dezfouli2019"
    lstm_path = os.path.join(base_path, "lstm_dezfouli2019.pkl")
    spice_path = os.path.join(base_path, "spice_dezfouli2019_l2_0_001.pkl")
    
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
        return
    
    # TODO: Load SPICE model (will be implemented next)
    print("SPICE model loading will be implemented next...")
    
    # TODO: Compare models (will be implemented after both models are loaded)
    print("Model comparison will be implemented after both models are loaded...")


if __name__ == "__main__":
    main()
