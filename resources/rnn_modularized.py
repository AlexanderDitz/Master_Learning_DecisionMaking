import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import pysindy as ps
import numpy as np


class GRUModule(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        
        self.gru_in = nn.GRU(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(1, 1)
       
    def forward(self, inputs):
        n_actions = inputs.shape[1]
        inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2]).unsqueeze(0)
        next_state = self.gru_in(inputs[..., 1:], inputs[..., :1])[1].view(-1, n_actions, 1)
        next_state = self.dropout(next_state)
        next_state = self.linear_out(next_state)
        return next_state
    
    
class BaseRNN(nn.Module):
    
    init_value = {}
    rnn_module = []
    equation = []
    control_signal = []
    input_configuration = {}
    state_configuration = {}
    action_assignment = {}
    module_activation = {}
    
    def __init__(
        self, 
        n_actions: int, 
        hidden_size: int,
        embedding_size: int,
        n_participants: int,
        dropout: float,
        device=torch.device('cpu'),
        ):
        super(BaseRNN, self).__init__()
        
        # define general network parameters
        self.device = device
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_participants = n_participants
        self.dropout = dropout
        
        # set up the participant-embedding layer
        self.setup_participant_embedding()
        self.setup_scaling_modules()
        self.setup_rnn_modules()
        
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.recording = {key: [] for key in self.rnn_modules+self.control_signals}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.submodules_sindy = dict()
        self.betas = torch.nn.ModuleDict()
        
        self.state = self.set_initial_state()
        
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs, prev_state, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        actions = inputs[:, :, :self.n_actions].float()
        rewards = inputs[:, :, self.n_actions:2*self.n_actions].float()
        participant_ids = inputs[0, :, -1:].int()
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_ids[:, 0])
        
        # update the control state w.r.t. embeddings and indeces
        self.control_state['participant_embedding'] = participant_embedding
        self.control_state['participant_index'] = participant_ids
        
        # set up memory state
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # setup action value memory
        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)
        
        return (actions, rewards), logits, timesteps
    
    def post_forward_pass(self, logits, batch_first):
        # add model dim again and set state
        # self.set_state(*args)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            
        return logits
    
    def set_initial_state(self, batch_size=1):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
                
        for key in self.recording.keys():
            self.recording[key] = []
        
        # state dimensions: (habit_state, value_state, habit, value)
        # dimensions of states: (batch_size, substate, hidden_size)
        # self.set_state(*[init_value + torch.zeros([batch_size, self._n_actions], dtype=torch.float, device=self.device) for init_value in self.init_values])
        
        state = {key: torch.full(size=[batch_size, self.n_actions], fill_value=self.init_values[key], dtype=torch.float32, device=self.device) for key in self.init_values}
        
        self.set_state(state)
        return self.get_state()
        
    def set_state(self, state_dict):
        """this method sets the latent variables
        
        Args:
            state (Dict[str, torch.Tensor]): hidden state
        """
        
        # self._state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
        self.state = state_dict
      
    def get_state(self, detach=False):
        """this method returns the memory state
        
        Returns:
            Dict[str, torch.Tensor]: Dict of latent variables corresponding to the memory state
        """
        
        state = self.state
        if detach:
            state = {key: state[key].detach() for key in state}

        return state
    
    def set_device(self, device: torch.device): 
        self.device = device
        
    def record_signal(self, key, old_value, new_value: Optional[torch.Tensor] = None):
        """appends a new timestep sample to the recording. A timestep sample consists of the value at timestep t-1 and the value at timestep t

        Args:
            key (str): recording key to which append the sample to
            old_value (_type_): value at timestep t-1 of shape (batch_size, feature_dim)
            new_value (_type_): value at timestep t of shape (batch_size, feature_dim)
        """
        
        if new_value is None:
            new_value = torch.zeros_like(old_value) - 1
        
        old_value = old_value.view(-1, 1, old_value.shape[-1]).clone().detach().cpu().numpy()
        new_value = new_value.view(-1, 1, new_value.shape[-1]).clone().detach().cpu().numpy()
        sample = np.concatenate([old_value, new_value], axis=1)
        self.recording[key].append(sample)
        
    def get_recording(self, key):
        return self.recording[key]
    
    def setup_rnn_modules(self):
        """This method sets up the standard RNN-modules"""
        
        for key in self.rnn_modules:
            # GRU network
            module = GRUModule(input_size=len(self.input_configuration[key])+self.embedding_size, hidden_size=self.hidden_size, dropout=self.dropout)
            self.submodules_rnn[key] = module
            
    def setup_eq_module(self, key: str, equation: Callable):
        if key in self.equations:
            self.submodules_eq[key] = equation
        else:
            raise ValueError(f"Error setting up an equation module with the key " + key + ". Valid keys are " + self.equations)
        
    def setup_scaling_modules(self):
        for key in self.state:
            if 'x_value' in key:
                self.betas[key] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())
                
    def setup_participant_embedding(self):
        if self.embedding_size > 0:
            self.participant_embedding = torch.nn.Embedding(num_embeddings=self.n_participants, embedding_dim=self.embedding_size)
        else:
            self.embedding_size = 1
            self.participant_embedding = lambda participant_id: torch.ones_like(participant_id, device=participant_id.device, dtype=torch.float).view(-1, 1)
    
    def call_module(self, key_module: str):
        """Used to call a submodule of the RNN. Can be either: 
            1. RNN-module (saved in 'self.submodules_rnn')
            2. SINDy-module (saved in 'self.submodules_sindy')
            3. hard-coded equation (saved in 'self.submodules_eq')

        Args:
            key_module (str): _description_
            key_state (str): _description_
            action (torch.Tensor, optional): _description_. Defaults to None.
            inputs (Union[torch.Tensor, Tuple[torch.Tensor]], optional): _description_. Defaults to None.
            participant_embedding (torch.Tensor, optional): _description_. Defaults to None.
            participant_index (torch.Tensor, optional): _description_. Defaults to None.
            activation_rnn (Callable, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        record_signal = False
        
        # get value from memory state
        value = self.get_state()[self.state_configuration[key_module]].unsqueeze(-1)
        
        # get action from control state
        action = self.control_state['c_action'].unsqueeze(-1)
        
        # get inputs from control state
        if key_module in self.input_configuration or len(self.input_configuration[key_module]) > 0:
            inputs = torch.concatenate([self.control_state[control_input] for control_input in self.input_configuration[key_module]], dim=-1)
        else:
            inputs = torch.zeros((*value.shape[:-1], 0), dtype=torch.float32, device=value.device)
        # TODO: remove unnecessary if condition
        if inputs.dim()==2:
            inputs = inputs.unsqueeze(-1)
        
        # get participant embedding from control state
        participant_embedding = self.control_state['participant_embedding'].unsqueeze(1).repeat(1, value.shape[1], 1)
        
        # get participant id from control state
        participant_index = self.control_state['participant_id']
        
        if key_module in self.submodules_sindy.keys():                
            # sindy module
            
            # convert to numpy
            value = value.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            
            if inputs.shape[-1] == 0:
                # create dummy control inputs
                inputs = torch.zeros((*inputs.shape[:-1], 1))
            
            next_value = np.zeros_like(value)
            for index_batch in range(value.shape[0]):
                sindy_model = self.submodules_sindy[key_module][participant_index[index_batch].item()] if isinstance(self.submodules_sindy[key_module], dict) else self.submodules_sindy[key_module]
                next_value[index_batch] = np.concatenate(
                    [sindy_model.predict(value[index_batch, index_action], inputs[index_batch, index_action]) for index_action in range(self.n_actions)], 
                    axis=0,
                    )
            next_value = torch.tensor(next_value, dtype=torch.float32, device=self.device)

        elif key_module in self.submodules_rnn.keys():
            # rnn module
            record_signal = True if not self.training else False
            
            # Linear handling
            inputs = torch.concat((value, inputs, participant_embedding), dim=-1)
            update_value = self.submodules_rnn[key_module](inputs)
            
            next_value = value + update_value
            
            if key_module in self.module_activation:
                if isinstance(self.module_activation[key_module], callable):
                    next_value = self.module_activation[key_module](next_value)
                else:
                    raise TypeError("Given activation function for RNN module " + key_module + " is not a valid callable")
            
        elif key_module in self.submodules_eq.keys():
            # hard-coded equation
            next_value = self.submodules_eq[key_module](value.squeeze(-1), inputs).unsqueeze(-1)

        else:
            raise ValueError(f'Invalid module key {key_module}.')

        if key_module in self.action_assignment:
            # keep only actions necessary for that update and set others to zero
            action_assignment = action*self.action_assignment[key_module] + (1-action)*(1-self.action_assignment[key_module])
            next_value = next_value * action_assignment
        else:
            action_assignment = torch.ones_like(action)
        
        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        if record_signal:
            # record sample for SINDy training 
            self.record_signal(key_module, value.view(-1, self.n_actions), next_value.view(-1, self.n_actions))
        
        # TODO: Test necessary!!!
        # Problem may occure with learning rate
        # save next value in memory state
        self.state[self.state_configuration[key_module]][action_assignment == 1] = next_value[next_value == 1]
        
        return next_value.squeeze(-1)
    
    def compute_action_value(self):
        """Computes the scaled action value as a sum of the memory states which have 'x_value' in their name."""
        
        shape = list(self.state.keys())[0].shape
        device = list(self.state.keys())[0].device
        action_value = torch.zeros(shape[1:], device=device)
        for state in self.state:
            if 'x_value' in state:
                scaling_factor = self.betas[state] if isinstance(self.betas[state], nn.Parameter) else self.betas[state](self.control_state['participant_embedding'])
                action_value += scaling_factor * self.state[state]
        return action_value
    
    def integrate_sindy(self, modules: Dict[str, Iterable[ps.SINDy]]):
        # check that all provided modules find a place in the RNN
        checked = 0
        for m in modules:
            if m in self.submodules_rnn.keys():
                checked += 1
        assert checked == len(modules), f'Not all provided SINDy modules {tuple(modules.keys())} found a corresponding RNN module or vice versa.\nSINDy integration aborted.'
        
        # replace rnn modules with sindy modules
        self.submodules_sindy = modules


class RLRNN(BaseRNN):
    
    init_value = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
            'x_learning_rate_reward': 0.,
        }
    
    rnn_module = [
        'x_learning_rate_chosen',
        'x_learning_rate_not_chosen', 
        'x_value_reward_not_chosen', 
        'x_value_choice_chosen', 
        'x_value_choice_not_chosen',
        ]
    
    equations = [
        'x_value_reward_chosen',
    ]
    
    control_signal = [
        'c_action', 
        'c_reward', 
    ]
    
    input_configuration = {
        'x_learning_rate_reward': ['c_reward', 'x_value_reward'],
        'x_value_reward_chosen': ['c_reward', 'x_learning_rate_reward'],
    }
    
    state_configuration = {
        'x_learning_rate_chosen': 'x_learning_rate_reward',
        'x_learning_rate_not_chosen': 'x_learning_rate_reward',
        'x_value_reward_chosen': 'x_value_reward', 
        'x_value_reward_not_chosen': 'x_value_reward',  
        'x_value_choice_chosen': 'x_value_choice', 
        'x_value_choice_not_chosen': 'x_value_choice',
    }
    
    action_assignment = {
        'x_learning_rate_chosen': 1,
        'x_learning_rate_not_chosen': 0,
        'x_value_reward_chosen': 1,
        'x_value_reward_not_chosen': 0,
        'x_value_choice_chosen': 1,
        'x_value_choice_not_chosen': 0,
    }
    
    module_activation = {
        'x_learning_rate_chosen': torch.nn.functional.sigmoid,
        'x_learning_rate_not_chosen': torch.nn.functional.sigmoid,
        'x_value_choice_chosen': torch.nn.functional.sigmoid, 
        'x_value_choice_not_chosen': torch.nn.functional.sigmoid,        
    }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        hidden_size = 8,
        embedding_size = 8,
        dropout = 0.,
        device = torch.device('cpu'),
        **kwargs,
    ):
        
        super(RLRNN, self).__init__(n_actions=n_actions, hidden_size=hidden_size, dropout=dropout, n_participants=n_participants, embedding_size=embedding_size, device=device)
        
        # set up hard-coded equations
        self.setup_eq_module(key='x_value_reward_chosen', equation=lambda value, inputs: value + inputs[..., 1] * (inputs[..., 0] - value))
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards = inputs
        
        # compute here derived inputs, e.g. trials since last action switch
        
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # update the control state w.r.t current control signals
            self.control_state['c_action'] = action
            self.control_state['c_reward'] = reward
            
            # updates for x_value_reward
            self.call_module('x_learning_rate_chosen')
            self.call_module('x_learning_rate_not_chosen')
            # update learning rate also in control state since it is used in other modules as an input
            self.control_state['c_learning_rate'] = self.state['x_learning_rate']
            
            self.call_module('x_value_reward_chosen')
            self.call_module('x_value_reward_not_chosen')
            # update value_reward also in control state since it is used in other modules as an input
            self.control_state['c_value_reward'] = self.state['x_value_reward']
            
            # updates for x_value_choice
            self.call_module('x_value_choice_chosen')
            self.call_module('x_value_choice_not_chosen')
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('c_learning_rate', self.state['x_learning_rate'])
            self.record_signal('c_value_reward', self.state['x_value_reward'])
            
            # compute action values
            logits[timestep] = self.compute_action_value()
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()