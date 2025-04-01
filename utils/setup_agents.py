import sys
import os

from typing import List, Dict
from torch import device, load
import numpy as np
import torch
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn import RLRNN
from resources.bandits import AgentSpice, AgentNetwork, AgentQ
from resources.sindy_training import fit_model
from utils.convert_dataset import convert_dataset
from benchmarking.hierarchical_bayes_numpyro import rl_model


def setup_rnn(
    path_model,
    list_sindy_signals, 
    n_actions=2,
    counterfactual=False,
    device=device('cpu'),
) -> RLRNN:
    
    # get n_participants and hidden_size from state dict
    state_dict = torch.load(path_model, map_location=torch.device('cpu'))['model']
    
    participant_embedding_index = [i for i, s in enumerate(list(state_dict.keys())) if 'participant_embedding' in s]
    participant_embedding_bool = True if len(participant_embedding_index) > 0 else False
    n_participants = 0 if not participant_embedding_bool else state_dict[list(state_dict.keys())[participant_embedding_index[0]]].shape[0]
    
    key_hidden_size = [key for key in state_dict if 'x' in key.lower()][0]  # first key that contains the hidden_size
    hidden_size = state_dict[key_hidden_size].shape[0]
    
    key_embedding_size = [key for key in state_dict if 'embedding' in key.lower()]
    if len(key_embedding_size) > 0:
        embedding_size = state_dict[key_embedding_size[0]].shape[1]
    else:
        embedding_size = 0
        
    rnn = RLRNN(
        n_actions=n_actions, 
        hidden_size=hidden_size, 
        embedding_size=embedding_size,
        n_participants=n_participants, 
        list_signals=list_sindy_signals, 
        device=device, 
        counterfactual=counterfactual,
        )
    rnn.load_state_dict(state_dict)
    
    return rnn


def setup_agent_rnn(
    path_model,
    list_sindy_signals,
    n_actions=2,
    counterfactual=False,
    deterministic=True,
    device=device('cpu'),
    ) -> AgentNetwork:
    
    rnn = setup_rnn(path_model=path_model, list_sindy_signals=list_sindy_signals, device=device, n_actions=n_actions, counterfactual=counterfactual)
    agent = AgentNetwork(model_rnn=rnn, n_actions=n_actions, deterministic=deterministic)
    
    return agent


def setup_agent_spice(
    path_model: str,
    path_data: str,
    rnn_modules: List[str],
    control_parameters: List[str],
    sindy_library_polynomial_degree: int,
    sindy_library_setup: Dict[str, List],
    sindy_filter_setup: Dict[str, List],
    sindy_dataprocessing: Dict[str, List],
    n_trials = 1024,
    threshold = 0.05,
    regularization = 1,
    participant_id: int = None,
    deterministic = True,
    verbose: bool = False,
) -> AgentSpice:
    
    agent_rnn = setup_agent_rnn(path_model=path_model, list_sindy_signals=rnn_modules+control_parameters)
    dataset = convert_dataset(file=path_data)[0]
    
    agent_spice = fit_model(
        agent=agent_rnn,
        data=dataset,
        rnn_modules=rnn_modules,
        control_parameters=control_parameters,
        polynomial_degree=sindy_library_polynomial_degree,
        library_setup=sindy_library_setup,
        filter_setup=sindy_filter_setup,
        dataprocessing=sindy_dataprocessing,
        participant_id=participant_id,
        n_trials_off_policy=n_trials,
        optimizer_alpha=regularization,
        optimizer_threshold=threshold,
        deterministic=deterministic,
        verbose=verbose,
    )

    return agent_spice


def setup_agent_mcmc(
    path_model: str,
) -> List[AgentQ]:
    
    # setup mcmc agent
    with open(path_model, 'rb') as file:
        mcmc = pickle.load(file)
    
    n_sessions = mcmc.get_samples()[list(mcmc.get_samples().keys())[0]].shape[-1]
    
    agents = []
    
    for session in range(n_sessions):
        parameters = {
            'alpha_pos': 1,
            'alpha_neg': -1,
            'alpha_c': 1,
            'beta_c': 0,
            'beta_r': 1,
        }
        
        for param in parameters:
            if param in mcmc.get_samples():
                samples = mcmc.get_samples()[param]
                if len(samples.shape) == 2:
                    samples = samples[:, session]
                parameters[param] = np.mean(samples, axis=0)
        
        if np.mean(parameters['alpha_neg']) == -1:
            parameters['alpha_neg'] = parameters['alpha_pos']
            
        agents.append(AgentQ(
            alpha_reward=parameters['alpha_pos'],
            alpha_penalty=parameters['alpha_neg'],
            alpha_choice=parameters['alpha_c'],
            beta_reward=parameters['beta_r'],
            beta_choice=parameters['beta_c'],
        ))
    
    return agents
        
    
    


if __name__ == '__main__':
    
    setup_agent_spice(
        path_model = 'params/benchmarking/sugawara2021_143_4.pkl',
        path_data = 'data/sugawara2021_143_processed.csv',
    )