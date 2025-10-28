"""
Model and Dataset Loading Utilities

Provides functions to load the Dezfouli 2019 dataset and all supported models (LSTM, SPICE, RNN, GQL).
"""

import sys
import os
import torch
import pickle
import numpy as np
import pandas as pd
from benchmarking.benchmarking_lstm import RLLSTM, AgentLSTM
from benchmarking.benchmarking_dezfouli2019 import AgentGQL, setup_agent_gql, Dezfouli2019GQL
from utils.setup_agents import setup_agent_spice, setup_agent_rnn
from resources.bandits import AgentSpice, AgentNetwork
from resources.rnn import RLRNN_eckstein2022


def load_dezfouli_dataset():
    """
    Load the Dezfouli 2019 dataset using dezfouli2019.csv and join with diagnosis info from original_data.csv.
    
    Returns:
        dataset: Dictionary of participant data (participant_id -> DataFrame)
    """
    try:
        # Get the correct path relative to the analysis directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Load both CSV files
        import pandas as pd
        dezfouli_path = os.path.join(parent_dir, "data", "preprocessing", "dezfouli2019.csv")
        original_path = os.path.join(parent_dir, "data", "preprocessing", "original_data.csv")
        
        # Load main dataset
        df_main = pd.read_csv(dezfouli_path)
        print(f"✓ Loaded dezfouli2019.csv: {len(df_main)} trials")
        
        # Load diagnosis information
        df_diagnosis = pd.read_csv(original_path)
        print(f"✓ Loaded original_data.csv: {len(df_diagnosis)} trials")
        
        # Create a mapping from participant ID to diagnosis (one entry per participant)
        diagnosis_mapping = df_diagnosis[['ID', 'diag']].drop_duplicates()
        print(f"✓ Found diagnosis info for {len(diagnosis_mapping)} participants")
        
        # Join main dataset with diagnosis information
        df_merged = df_main.merge(
            diagnosis_mapping, 
            left_on='df_participant_id', 
            right_on='ID', 
            how='inner'  # Use inner join to keep only participants with diagnosis info
        )
        
        print(f"✓ Successfully joined datasets: {len(df_merged)} trials with diagnosis info")
        
        # Rename columns to standard format and create participant dictionary
        df_merged = df_merged.rename(columns={
            'df_choice': 'choice',
            'df_reward': 'reward', 
            'df_session': 'session',
            'diag': 'diagnosis'
        })
        
        # Group by participant ID and create dictionary
        dataset = {pid: group[['choice', 'reward', 'session', 'diagnosis']].reset_index(drop=True) 
                  for pid, group in df_merged.groupby('df_participant_id')}
        
        print(f"✓ Loaded Dezfouli dataset: {len(dataset)} participants")
        
        # Print diagnostic info
        if dataset:
            first_participant = list(dataset.keys())[0]
            print(f"  Sample participant {first_participant}: {len(dataset[first_participant])} trials")
            print(f"  Columns: {list(dataset[first_participant].columns)}")
            
            # Print diagnosis distribution
            diagnosis_counts = {}
            for participant_data in dataset.values():
                diagnosis = participant_data['diagnosis'].iloc[0]
                diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
            print(f"  Diagnosis distribution: {diagnosis_counts}")
        
        return dataset
    except Exception as e:
        print(f"✗ Failed to load Dezfouli dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_rewards_array(participant_data):
    """
    Convert participant data to two-armed bandit reward format.
    
    Args:
        participant_data: DataFrame with 'choice' and 'reward' columns
        
    Returns:
        rewards: Array of shape (n_trials, 2) for two-armed bandit
    """
    actions = participant_data['choice'].values.astype(int)
    rewards = participant_data['reward'].values
    
    # Create reward matrix for two-armed bandit
    rewards_matrix = np.zeros((len(actions), 2))
    rewards_matrix[np.arange(len(actions)), actions] = rewards
    
    return rewards_matrix


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
    
    try:
        # Import the GQL classes
        from benchmarking.benchmarking_dezfouli2019 import Dezfouli2019GQL, AgentGQL
        
        # Create a custom unpickler to handle the class import issue
        import pickle
        import sys
        
        # Add the class to the current module for unpickling
        current_module = sys.modules[__name__]
        setattr(current_module, 'Dezfouli2019GQL', Dezfouli2019GQL)
        
        # Try loading with torch.load first (in case it was saved with torch)
        try:
            all_models = torch.load(path_model, map_location=torch.device('cpu'), weights_only=False)
        except:
            # Fallback to pickle load
            with open(path_model, 'rb') as file:
                all_models = pickle.load(file)
        
        # Ensure all_models is a list
        if not isinstance(all_models, list):
            all_models = [all_models]
        
        # Create agents from loaded models
        agent_gql = []
        for model in all_models:
            agent_gql.append(AgentGQL(model=model, deterministic=True))
        
        # Calculate number of parameters from the first model
        model = all_models[0]
        n_parameters = 0
        for index_letter, letter in enumerate(model_config):
            if not letter.islower():
                n_parameters += 1 * model.d * model.d if letter == 'C' and index_letter == len(model_config)-1 else 1 * model.d
        
        print("GQL model loaded successfully!")
        print(f"  - Model type: {type(agent_gql[0])}")
        print(f"  - Number of participants: {len(agent_gql)}")
        print(f"  - Number of actions: {agent_gql[0]._n_actions}")
        print(f"  - Total parameters: {n_parameters}")
        
        return agent_gql, n_parameters
        
    except Exception as e:
        print(f"❌ Failed to load GQL model: {e}")
        import traceback
        traceback.print_exc()
        return None