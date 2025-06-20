import sys, os

import numpy as np
import torch
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import argparse
import pickle
from typing import List, Callable, Union, Dict, Tuple
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import split_data_along_sessiondim, reshape_data_along_participantdim
from resources.bandits import Agent, check_in_0_1_range
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session
from resources.rnn_utils import DatasetRNN


def gql_update_step(q_values, h_values, choice, reward, params, d=2):
    """
    GQL update function implementing the Generalized Q-Learning model from Dezfouli et al. 2019.
    
    Args:
        q_values: Current action values [..., n_actions, d]
        h_values: Current choice histories [..., n_actions, d]
        choice: Choice made (0 or 1) or one-hot encoded choice
        reward: Reward received (scalar or array)
        params: Dictionary containing learning parameters
        d: Number of different values/histories tracked per action
        
    Returns:
        Updated q_values, h_values, and action probabilities
    """
    n_actions = q_values.shape[-2]
    
    # Handle both scalar choice and one-hot encoded choice
    if isinstance(choice, (int, np.integer)) or choice.ndim == 0:
        choice_onehot = jnp.eye(n_actions)[choice]
    else:
        choice_onehot = choice
    
    # Ensure reward has proper shape for broadcasting
    if reward.ndim == q_values.ndim - 2:
        # Add last two dimensions to match q_values shape
        reward = jnp.expand_dims(jnp.expand_dims(reward, axis=-1), axis=-1)
    elif reward.ndim == q_values.ndim - 1 and reward.shape[-1] == n_actions:
        # Add last dimension for d
        reward = jnp.expand_dims(reward, axis=-1)
    
    # Broadcast reward to match q_values shape
    reward = jnp.broadcast_to(reward, q_values.shape)
    
    # Expand parameter dimensions to match q_values for broadcasting
    def expand_param(param, target_shape):
        if jnp.isscalar(param) or param.ndim == 0:
            return param
        else:
            # For parameters with shape (n_participants, d), we need to expand to (n_participants, n_actions, d)
            if param.shape == target_shape[:-2] + (target_shape[-1],):
                # Add the n_actions dimension
                param = jnp.expand_dims(param, axis=-2)
            # Add dimensions to match target shape
            while param.ndim < len(target_shape):
                param = jnp.expand_dims(param, axis=-1)
            return jnp.broadcast_to(param, target_shape)
    
    # Learning rates for Q-values (phi - shape: [..., d])
    phi = expand_param(params['phi'], q_values.shape)
    
    # Q-value updates
    # Expand choice_onehot to match q_values dimensions
    choice_expanded = jnp.expand_dims(choice_onehot, axis=-1)  # [..., n_actions, 1]
    choice_expanded = jnp.broadcast_to(choice_expanded, q_values.shape)  # [..., n_actions, d]
    
    # Update Q-values: Q_t = (1 - phi) * Q_{t-1} + phi * reward (for chosen action)
    q_update = phi * reward * choice_expanded
    q_values_new = (1 - phi) * q_values + q_update
    
    # Learning rates for choice histories (chi - shape: [..., d])
    chi = expand_param(params['chi'], h_values.shape)
    
    # History updates
    # For chosen action: H_t = (1 - chi) * H_{t-1} + chi
    # For other actions: H_t = (1 - chi) * H_{t-1}
    h_chosen_update = chi * choice_expanded
    h_decay = (1 - chi) * h_values
    h_values_new = h_decay + h_chosen_update
    
    # Compute action probabilities
    # Parameters for combining Q-values and histories
    beta = expand_param(params['beta'], q_values.shape)  # weights for Q-values
    kappa = expand_param(params['kappa'], h_values.shape)  # weights for histories
    
    # Linear combination of Q-values and histories
    q_weighted = jnp.sum(beta * q_values_new, axis=-1)  # [..., n_actions]
    h_weighted = jnp.sum(kappa * h_values_new, axis=-1)  # [..., n_actions]
    
    # Add interaction terms
    C = expand_param(params['C'], q_values.shape[:-2] + (d, d))
    # Compute interaction: H^T * C * Q for each action
    interaction = jnp.einsum('...ad,...de,...ae->...a', h_values_new, C, q_values_new)
    combined_values = q_weighted + h_weighted + interaction
    
    # Softmax to get action probabilities
    action_prob_0 = jax.nn.sigmoid(combined_values[..., 0] - combined_values[..., 1])
    
    return q_values_new, h_values_new, action_prob_0


class Agent_dezfouli2019(Agent):
    """An agent that runs GQL (Generalized Q-Learning) for the two-armed bandit task."""

    def __init__(
        self,
        n_actions: int = 2,
        d: int = 2,
        phi: Union[float, np.ndarray] = None,
        chi: Union[float, np.ndarray] = None,
        beta: Union[float, np.ndarray] = None,
        kappa: Union[float, np.ndarray] = None,
        C: np.ndarray = None,
    ):
        
        self._n_actions = n_actions
        self._d = d  # Number of different values/histories per action
        self._q_init = 0.5
        self._h_init = 0.0
        
        super().__init__(n_actions=n_actions)
          
        # Default parameters if not provided
        if phi is None:
            phi = np.full(d, 0.1)
        if chi is None:
            chi = np.full(d, 0.1)
        if beta is None:
            beta = np.full(d, 1.0)
        if kappa is None:
            kappa = np.full(d, 0.1)
        if C is None:
            C = np.zeros((d, d))
        
        # Store parameters in format expected by shared update function
        self._params = {
            'phi': np.array(phi) if np.isscalar(phi) else phi,
            'chi': np.array(chi) if np.isscalar(chi) else chi,
            'beta': np.array(beta) if np.isscalar(beta) else beta,
            'kappa': np.array(kappa) if np.isscalar(kappa) else kappa,
            'C': C,
        }
                
        # Validation
        if hasattr(phi, '__iter__'):
            for p in phi:
                check_in_0_1_range(p, 'phi')
        else:
            check_in_0_1_range(phi, 'phi')
            
        if hasattr(chi, '__iter__'):
            for c in chi:
                check_in_0_1_range(c, 'chi')
        else:
            check_in_0_1_range(chi, 'chi')

    def new_sess(self, **kwargs):
        """Initialize a new session."""
        super().new_sess()
        # Initialize Q-values and choice histories
        self._state['x_value_reward'] = jnp.full((self._n_actions, self._d), self._q_init)
        self._state['x_value_choice'] = jnp.full((self._n_actions, self._d), self._h_init)

    def update(self, choice: int, reward: np.ndarray, *args, **kwargs):
        """Update the agent after one step using shared update logic."""
        # Extract the reward for the chosen action, or use scalar reward
        if isinstance(reward, np.ndarray) and reward.size > 1:
            reward_value = reward[choice]
        else:
            reward_value = reward.item() if hasattr(reward, 'item') else reward
        
        # Use shared update function
        q_values_new, h_values_new, _ = gql_update_step(
            self._state['x_value_reward'],
            self._state['x_value_choice'],
            choice,
            reward_value,
            self._params,
            self._d
        )
        
        # Update state
        self._state['x_value_reward'] = q_values_new
        self._state['x_value_choice'] = h_values_new

    @property
    def q(self):
        """Return current action values (weighted combination)."""
        beta = self._params['beta']
        kappa = self._params['kappa']
        C = self._params['C']
        
        q_weighted = jnp.sum(beta * self._state['x_value_reward'], axis=-1)
        h_weighted = jnp.sum(kappa * self._state['x_value_choice'], axis=-1)
        
        # Add interaction terms
        interaction = jnp.einsum('ad,de,ae->a', self._state['x_value_choice'], C, self._state['x_value_reward'])
        return q_weighted + h_weighted + interaction


def setup_agent_mcmc(path_model: str) -> Tuple[List[Agent_dezfouli2019], np.ndarray]:
    """Setup MCMC agents using participant-level parameters."""
    
    with open(path_model, 'rb') as file:
        mcmc = pickle.load(file)
    
    n_participants = mcmc.get_samples()['beta'].shape[-1]
    d = mcmc.get_samples()['beta'].shape[-2]
    
    agents = []
    
    for participant in range(n_participants):
        
        # Default parameters
        parameters = {
            'phi': np.full(d, 0.1),
            'chi': np.full(d, 0.1),
            'beta': np.full(d, 1.0),
            'kappa': np.full(d, 0.1),
            'C': np.zeros((d, d)),
        }
        
        # Extract parameters from MCMC samples (these are now participant-level)
        for param in parameters:
            if param in mcmc.get_samples():
                parameters[param] = mcmc.get_samples()[param][..., participant].mean(axis=0)
            elif param == 'C':
                parameters['C'] = mcmc.get_samples()['C_vec'][..., participant].reshape(-1, d, d).mean(axis=0)    
        
        agents.append(Agent_dezfouli2019(
            d=d,
            phi=parameters['phi'],
            chi=parameters['chi'],
            beta=parameters['beta'],
            kappa=parameters['kappa'],
            C=parameters['C'],
        ))
    
    return agents


def gql_model(choice, reward, d=2):
    """
    A GQL model with participant-level parameters shared across sessions.
    Each participant has one set of parameters used across all their sessions.
    
    Args:
        choice: Shape (time, n_sessions, 2) - all sessions concatenated
        reward: Shape (time, n_sessions, 1) - all sessions concatenated  
        participant_mapping: Shape (n_sessions,) - maps each session to participant index
        d: Number of different values/histories per action
    """
    
    # Hierarchical priors for learning rates (phi)
    phi_mean = numpyro.sample("phi_mean", dist.Beta(1, 1))
    phi_kappa = numpyro.sample("phi_kappa", dist.HalfNormal(1.0))
    
    # Hierarchical priors for choice learning rates (chi)
    chi_mean = numpyro.sample("chi_mean", dist.Beta(1, 1))
    chi_kappa = numpyro.sample("chi_kappa", dist.HalfNormal(1.0))
    
    # Hierarchical priors for Q-value weights (beta)
    beta_mean = numpyro.sample("beta_mean", dist.Normal(0, 1))
    beta_sigma = numpyro.sample("beta_sigma", dist.HalfNormal(1.0))
    
    # Hierarchical priors for choice weights (kappa)
    kappa_mean = numpyro.sample("kappa_mean", dist.Normal(0, 1))
    kappa_sigma = numpyro.sample("kappa_sigma", dist.HalfNormal(3.0))
    
    # Hierarchical priors for interaction matrix (C)
    C_mean = numpyro.sample("C_mean", dist.Normal(0, 1))
    C_sigma = numpyro.sample("C_sigma", dist.HalfNormal(3.0))
    
    # Participant-level parameters
    n_participants = choice.shape[1]
    with numpyro.plate("participants", n_participants):
        with numpyro.plate("d_phi", d):
            phi_raw = numpyro.sample("phi", dist.Beta(phi_mean * phi_kappa, (1 - phi_mean) * phi_kappa))
        phi = phi_raw.T  # Shape: (n_participants, d)
        
        with numpyro.plate("d_chi", d):
            chi_raw = numpyro.sample("chi", dist.Beta(chi_mean * chi_kappa, (1 - chi_mean) * chi_kappa))
        chi = chi_raw.T
        
        with numpyro.plate("d_beta", d):
            beta_raw = numpyro.sample("beta", dist.Normal(beta_mean, beta_sigma))
        beta = beta_raw.T
        
        with numpyro.plate("d_kappa", d):
            kappa_raw = numpyro.sample("kappa", dist.Normal(kappa_mean, kappa_sigma)) * 15
        kappa = kappa_raw.T
        
        # Interaction matrix
        with numpyro.plate("d_C", d * d):
            C_vec_raw = numpyro.sample("C_vec", dist.Normal(C_mean, C_sigma))
        C_vec = C_vec_raw.T
        C = C_vec.reshape((n_participants, d, d))
    
    # Map participant parameters to sessions
    n_sessions = choice.shape[2]
    session_phi = jnp.expand_dims(phi, 1).repeat(n_sessions, 1)       # Shape: (n_sessions, d)
    session_chi = jnp.expand_dims(chi, 1).repeat(n_sessions, 1)       # Shape: (n_sessions, d)
    session_beta = jnp.expand_dims(beta, 1).repeat(n_sessions, 1)     # Shape: (n_sessions, d)
    session_kappa = jnp.expand_dims(kappa, 1).repeat(n_sessions, 1)   # Shape: (n_sessions, d)
    session_C = jnp.expand_dims(C, 1).repeat(n_sessions, 1)        # Shape: (n_sessions, d, d)
    
    def update(carry, x):
        q_values, h_values = carry
        ch, rw = x[..., :2], x[..., 2]
        
        # Create parameter dict for sessions
        params = {
            'phi': session_phi,
            'chi': session_chi,
            'beta': session_beta,
            'kappa': session_kappa,
            'C': session_C,
        }
        
        # Use shared update function
        q_values_new, h_values_new, action_prob_0 = gql_update_step(q_values, h_values, ch, rw, params, d)
        
        # # Ensure action_prob_0 has the right shape
        # if action_prob_0.ndim > 1:
        #     action_prob_0 = action_prob_0.reshape(-1)
        
        return (q_values_new, h_values_new), action_prob_0

    # Initialize Q-values and choice histories (reset for each session)
    q_values = jnp.full((n_participants, n_sessions, 2, d), 0.5)
    h_values = jnp.zeros((n_participants, n_sessions, 2, d))
    
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    
    ys = jnp.zeros((choice.shape[0]-1, n_participants, n_sessions))
    carry = (q_values, h_values)
    for i in range(len(choice)-1):
        carry, y = update(carry, xs[i])
        ys = ys.at[i].set(y)

    # Likelihood
    next_choice_0 = choice[1:, ..., 0]
    valid_mask = (next_choice_0 >= 0) & (next_choice_0 <= 1)
    
    with numpyro.handlers.mask(mask=valid_mask):
        with numpyro.plate("participants", choice.shape[1], dim=-2):
            with numpyro.plate("sessions", choice.shape[2], dim=-1):
                with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-3):
                    numpyro.sample("obs", dist.Bernoulli(probs=ys), obs=next_choice_0)


def fit_mcmc(file: str, num_samples: int, num_warmup: int, num_chains: int, output_dir: str, checkpoint: bool, train_test_ratio: list = [3, 6, 9], d: int = 2):
    # Set output file
    output_file = os.path.join(output_dir, f'mcmc_dezfouli2019_gql_multi_session_d{d}.nc')
    
    # Get and prepare the data
    dataset = convert_dataset(file)[0]
    dataset = reshape_data_along_participantdim(split_data_along_sessiondim(dataset=dataset, list_test_sessions=train_test_ratio)[0])
        
    # Extract choices and rewards from the reshaped dataset
    xs = dataset.xs
    choices = xs[..., :2]  # First 2 features are choices
    rewards = torch.max(xs[..., 2:4], dim=-1)[0].unsqueeze(-1)  # Max of reward features
    
    print(f"Reshaped dataset: {xs.shape[0]} participants, {xs.shape[1]} participant-sessions, {xs.shape[2]} trials")
    
    # Run the model
    numpyro.set_host_device_count(num_chains)
    print(f'Number of devices: {jax.device_count()}')
    kernel = infer.NUTS(gql_model)
    if checkpoint and num_warmup > 0:
        print(f'Checkpoint was set but num_warmup>0 ({num_warmup}). Setting num_warmup=0.')
        num_warmup = 0
    mcmc = infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    print('Initialized MCMC model.')
    
    if checkpoint:
        with open(output_file, 'rb') as file:
            checkpoint = pickle.load(file)
        mcmc.post_warmup_state = checkpoint.last_state
        rng_key = mcmc.post_warmup_state.rng_key
        print('Checkpoint loaded.')
    else:
        rng_key = jax.random.PRNGKey(0)
        
    # Convert to JAX arrays and transpose to (time, batch)
    choice_jax = jnp.array(choices.numpy()).swapaxes(0, 2).swapaxes(1, 2)
    reward_jax = jnp.array(rewards.numpy()).swapaxes(0, 2).swapaxes(1, 2)
    
    mcmc.run(rng_key, choice=choice_jax, reward=reward_jax, d=d)
    
    with open(output_file, 'wb') as file:
        pickle.dump(mcmc, file)
    
    return mcmc


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Performs hierarchical bayesian parameter inference with numpyro for GQL model with participant-grouped sessions.')

    parser.add_argument('--file', type=str, default='data/dezfouli2019/dezfouli2019.csv', help='Dataset of a 2-armed bandit task with columns (session, choice, reward)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warmup samples (additional)')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    parser.add_argument('--output_dir', type=str, default='benchmarking/params', help='Output directory')
    parser.add_argument('--checkpoint', action='store_true', help='Whether to load the specified output file as a checkpoint')
    parser.add_argument('--d', type=int, default=2, help='Number of different values/histories per action')

    args = parser.parse_args()

    # jax.config.update('jax_disable_jit', True)
    
    mcmc = fit_mcmc(args.file, args.num_samples, args.num_warmup, args.num_chains, args.output_dir, args.checkpoint, d=args.d)

    # Example usage with participant-level agents
    # agents, participant_mapping = setup_agent_mcmc(os.path.join(args.output_dir, f'mcmc_dezfouli2019_gql_multi_session_d{args.d}.nc'))
    # print(f"Created {len(agents)} participant-level agents")
    # print(f"Participant mapping: {participant_mapping}")
    
    # To get participant-grouped data for analysis:
    # participant_dataset, participant_mapping = prepare_participant_data(convert_dataset(args.file)[0])
    # experiment = participant_dataset.xs[0]  # First session of first participant
    # fig, axs = plot_session(agents={'benchmark': agents[0]}, experiment=experiment)
    # plt.show()