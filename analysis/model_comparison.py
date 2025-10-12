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
from benchmarking.benchmarking_dezfouli2019 import AgentGQL, setup_agent_gql, Dezfouli2019GQL
from utils.setup_agents import setup_agent_spice, setup_agent_rnn
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentSpice, AgentNetwork
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
        
        print("\nModel comparison metrics will be implemented next...")
    else:
        print("❌ No models could be loaded. Cannot proceed with comparison.")
        for model_name, model in models_loaded.items():
            if model is None:
                print(f"  - {model_name} model failed to load")


if __name__ == "__main__":
    main()
