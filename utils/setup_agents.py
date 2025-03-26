import sys
import os

from typing import List, Dict
from torch import device, load
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn import RLRNN
from resources.bandits import AgentSindy, AgentNetwork, AgentQ
from resources.sindy_training import fit_model
from utils.convert_dataset import convert_dataset


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
    device=device('cpu'),
    ) -> AgentNetwork:
    
    rnn = setup_rnn(path_model=path_model, list_sindy_signals=list_sindy_signals, device=device, n_actions=n_actions, counterfactual=counterfactual)
    agent = AgentNetwork(model_rnn=rnn, n_actions=n_actions, deterministic=True)
    
    return agent


def setup_agent_sindy(
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
) -> AgentSindy:
    
    agent_rnn = setup_agent_rnn(path_model=path_model, list_sindy_signals=rnn_modules+control_parameters)
    dataset = convert_dataset(file=path_data)[0]
    
    agent_sindy = fit_model(
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
    )

    return agent_sindy


def setup_benchmark_q_agent(
    parameters,
    **kwargs
) -> AgentQ:
    
    class AgentBenchmark(AgentQ):
        
        def __init__(self, parameters, n_actions = 2):
            super().__init__(n_actions, 0, 0)
            
            self._parameters = parameters
            
        def update(self, a, r, *args):
            # q, c = update_rule(self._q, self._c, a, r)
            ch = np.eye(2)[a]
            r = r[0]
            
            # Compute prediction errors for each outcome
            rpe = (r - self._q) * ch
            cpe = ch - self._c
            
            # Update values
            lr = np.where(r > 0.5, self._parameters['alpha_pos'], self._parameters['alpha_neg'])
            self._q += lr * rpe
            self._c += self._parameters['alpha_c'] * cpe
            
        @property
        def q(self):
            return self._parameters['beta_r'] * self._q + self._parameters['beta_c'] * self._c
        
    return AgentBenchmark(parameters)
    
    
    


if __name__ == '__main__':
    
    setup_agent_sindy(
        path_model = 'params/benchmarking/sugawara2021_143_4.pkl',
        path_data = 'data/sugawara2021_143_processed.csv',
    )