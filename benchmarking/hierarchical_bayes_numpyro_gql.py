import sys, os

import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import argparse
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim


def gql_model(choice, reward, hierarchical, d=2):
    """
    Complete Generalized Q-learning model implementation based on Dezfouli et al. (2019)
    
    This implements the full GQL model with:
    - Multiple Q-values per action (d components) with different learning rates (Phi)
    - Action history tracking (d components) with different learning rates (Psi)  
    - Q-value combination weights (B)
    - Action history combination weights (K)
    - Q-value x action history interaction matrix (C)
    
    Args:
        choice (jnp.ndarray): Shape (T, N, 2) - binary choices over time
        reward (jnp.ndarray): Shape (T, N) - binary rewards over time  
        hierarchical (int): Whether to use hierarchical Bayesian inference
        d (int): Number of Q-value/history components per action (default=2 as in paper)
    """
    
    n_participants = choice.shape[1]
    
    if hierarchical == 1:
        # Group-level hyperparameters for GQL model
        # Q-value learning rates (Phi) - constrained to [0,1]
        phi_mean = numpyro.sample("phi_mean", dist.Beta(2, 2).expand([d]))
        phi_kappa = numpyro.sample("phi_kappa", dist.HalfNormal(1.0).expand([d]))
        
        # Action history learning rates (Psi) - constrained to [0,1]
        psi_mean = numpyro.sample("psi_mean", dist.Beta(2, 2).expand([d]))
        psi_kappa = numpyro.sample("psi_kappa", dist.HalfNormal(1.0).expand([d]))
        
        # Q-value weights (B) - can be positive or negative
        b_mean = numpyro.sample("b_mean", dist.Normal(0, 1).expand([d]))
        b_std = numpyro.sample("b_std", dist.HalfNormal(1.0).expand([d]))
        
        # Action history weights (K) - can be positive or negative
        k_mean = numpyro.sample("k_mean", dist.Normal(0, 1).expand([d]))
        k_std = numpyro.sample("k_std", dist.HalfNormal(1.0).expand([d]))
        
        # Interaction matrix (C) - small values as in paper
        c_mean = numpyro.sample("c_mean", dist.Normal(0, 0.1).expand([d, d]))
        c_std = numpyro.sample("c_std", dist.HalfNormal(0.1).expand([d, d]))
        
        # Individual-level parameters
        with numpyro.plate("participants", n_participants):
            phi = numpyro.sample("phi", 
                dist.Beta(phi_mean * phi_kappa, (1 - phi_mean) * phi_kappa).expand([d]).to_event(1))
            psi = numpyro.sample("psi",
                dist.Beta(psi_mean * psi_kappa, (1 - psi_mean) * psi_kappa).expand([d]).to_event(1))
            b = numpyro.sample("b", 
                dist.Normal(b_mean, b_std).expand([d]).to_event(1))
            k = numpyro.sample("k",
                dist.Normal(k_mean, k_std).expand([d]).to_event(1))
            c = numpyro.sample("c",
                dist.Normal(c_mean, c_std).expand([d, d]).to_event(2))
            
    else:
        # Non-hierarchical version - simpler priors
        phi = numpyro.sample("phi", dist.Beta(2, 2).expand([d]))
        psi = numpyro.sample("psi", dist.Beta(2, 2).expand([d]))
        b = numpyro.sample("b", dist.Normal(0, 1).expand([d]))
        k = numpyro.sample("k", dist.Normal(0, 1).expand([d]))
        c = numpyro.sample("c", dist.Normal(0, 0.1).expand([d, d]))
    
    def gql_update(carry, x):
        """GQL update function implementing the complete model from Dezfouli et al."""
        q_values, h_values = carry
        ch, rw = x[:, :2], x[:, 2][:, None]
        
        # Update Q-values: Q_t(a) = Q_{t-1}(a) + Φ ⊙ (r - Q_{t-1}(a)) for taken action
        for action in range(2):
            action_mask = ch[:, action:action+1]  # Keep dims: (n_participants, 1)
            prediction_errors = rw - q_values[:, action, :]  # (n_participants, d)
            
            if hierarchical == 1:
                # phi has shape (n_participants, d)
                update = phi * prediction_errors * action_mask
            else:
                # phi has shape (d,), broadcast to (n_participants, d)
                update = phi[None, :] * prediction_errors * action_mask
                
            q_values = q_values.at[:, action, :].add(update)
        
        # Update action histories: H_t(a) = H_{t-1}(a) + Ψ ⊙ (1 - H_{t-1}(a)) for taken action
        #                         H_t(other) = H_{t-1}(other) - Ψ ⊙ H_{t-1}(other) for other actions
        for action in range(2):
            action_mask = ch[:, action:action+1]  # (n_participants, 1)
            not_action_mask = 1 - action_mask     # (n_participants, 1)
            
            if hierarchical == 1:
                # Update for taken actions: increase history
                taken_update = psi * (1 - h_values[:, action, :]) * action_mask
                # Update for non-taken actions: decrease history  
                not_taken_update = -psi * h_values[:, action, :] * not_action_mask
            else:
                # psi has shape (d,), broadcast to (n_participants, d)
                taken_update = psi[None, :] * (1 - h_values[:, action, :]) * action_mask
                not_taken_update = -psi[None, :] * h_values[:, action, :] * not_action_mask
                
            h_values = h_values.at[:, action, :].add(taken_update + not_taken_update)
        
        # Compute action probabilities: P(a) = softmax(B·Q(a) + K·H(a) + Σ C_ij * Q(a)_i * H(a)_j)
        action_values = jnp.zeros((n_participants, 2))
        
        for action in range(2):
            if hierarchical == 1:
                # Linear terms: B·Q(a) + K·H(a)
                linear_term = jnp.sum(b * q_values[:, action, :], axis=1) + jnp.sum(k * h_values[:, action, :], axis=1)
                
                # Interaction term: sum(C_ij * Q(a)_i * H(a)_j) using einsum for efficiency
                interaction_term = jnp.einsum('nij,ni,nj->n', c, q_values[:, action, :], h_values[:, action, :])
            else:
                # Linear terms with broadcasting
                linear_term = jnp.dot(q_values[:, action, :], b) + jnp.dot(h_values[:, action, :], k)
                
                # Interaction terms using einsum
                interaction_term = jnp.einsum('ij,ni,nj->n', c, q_values[:, action, :], h_values[:, action, :])
            
            action_values = action_values.at[:, action].set(linear_term + interaction_term)
        
        # Convert to probabilities using softmax (equivalent to sigmoid for 2 actions)
        action_diff = action_values[:, 0] - action_values[:, 1]
        action_prob_option_0 = jax.nn.sigmoid(action_diff)
        
        return (q_values, h_values), action_prob_option_0
    
    # Initialize values as in the paper
    q_values = jnp.full((n_participants, 2, d), 0.5)  # Initial Q-values at 0.5
    h_values = jnp.zeros((n_participants, 2, d))      # Initial action histories at 0
    
    # Prepare data for scanning
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    
    # Run the model forward using jax.lax.scan for efficiency
    carry_init = (q_values, h_values)
    _, ys = jax.lax.scan(gql_update, carry_init, xs)
    
    # Likelihood
    next_choice_0 = choice[1:, :, 0]
    valid_mask = (next_choice_0 >= 0) & (next_choice_0 <= 1)
    
    if hierarchical == 1:
        with numpyro.handlers.mask(mask=valid_mask):
            with numpyro.plate("participants", n_participants, dim=-1):
                with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
                    numpyro.sample("obs", dist.Bernoulli(probs=ys), obs=next_choice_0)
    else:
        with numpyro.handlers.mask(mask=valid_mask.flatten()):
            numpyro.sample("obs", dist.Bernoulli(probs=ys.flatten()), obs=next_choice_0.flatten())


def rl_model(model, choice, reward, hierarchical):
    """
    Standard reinforcement learning model with configurable components.
    
    Args:
        model (list of int): Binary vector indicating which parameters to include:
            - model[0]: Include individual positive learning rate (alpha_pos)
            - model[1]: Include individual negative learning rate (alpha_neg)
            - model[2]: Include counterfactual positive learning rate (alpha_cf_pos)
            - model[3]: Include counterfactual negative learning rate (alpha_cf_neg)
            - model[4]: Include choice perseverance (alpha_ch)
            - model[5]: Include reward sensitivity scaling (beta_ch)
            - model[6]: Include inverse temperature for reward sensitivity (beta_r)
            - model[7]: Include counterfactual sensitivity scaling (beta_cf)
        choice, reward, hierarchical: Same as gql_model
    """
    
    def scaled_beta(a, b, low, high):
        return dist.TransformedDistribution(
            dist.Beta(a, b),
            dist.transforms.AffineTransform(0, high - low)
        )
    
    beta_scaling = 15
    n_participants = choice.shape[1]
    
    if hierarchical == 1:
        # Group-level hyperparameters
        alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Beta(1, 1)) if model[0]==1 else 1
        alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Beta(1, 1)) if model[1]==1 else -1
        alpha_cf_pos_mean = numpyro.sample("alpha_cf_pos_mean", dist.Beta(1, 1)) if model[2]==1 else 0
        alpha_cf_neg_mean = numpyro.sample("alpha_cf_neg_mean", dist.Beta(1, 1)) if model[3]==1 else 0
        alpha_ch_mean = numpyro.sample("alpha_ch_mean", dist.Beta(1, 1)) if model[4]==1 else 1
        beta_ch_mean = numpyro.sample("beta_ch_mean", dist.Beta(1, 1)) if model[5]==1 else 0
        beta_r_mean = numpyro.sample("beta_r_mean", dist.Beta(1, 1)) if model[6]==1 else 1
        
        # Individual-level variation
        alpha_pos_kappa = numpyro.sample("alpha_pos_kappa", dist.HalfNormal(1.0)) if model[0]==1 else 0
        alpha_neg_kappa = numpyro.sample("alpha_neg_kappa", dist.HalfNormal(1.0)) if model[1]==1 else 0
        alpha_cf_pos_kappa = numpyro.sample("alpha_cf_pos_kappa", dist.HalfNormal(1.0)) if model[2]==1 else 0
        alpha_cf_neg_kappa = numpyro.sample("alpha_cf_neg_kappa", dist.HalfNormal(1.0)) if model[3]==1 else 0
        alpha_ch_kappa = numpyro.sample("alpha_ch_kappa", dist.HalfNormal(1.0)) if model[4]==1 else 0
        beta_ch_kappa = numpyro.sample("beta_ch_kappa", dist.HalfNormal(1.0)) if model[5]==1 else 0
        beta_r_kappa = numpyro.sample("beta_r_kappa", dist.HalfNormal(1.0)) if model[6]==1 else 0
        
        # Individual-level parameters
        with numpyro.plate("participants", n_participants):
            if model[0]:
                alpha_pos = numpyro.sample("alpha_pos", dist.Beta(alpha_pos_mean * alpha_pos_kappa, (1 - alpha_pos_mean) * alpha_pos_kappa))[:, None]
            else:
                alpha_pos = jnp.full((n_participants, 1), 1.0)

            if model[1]:
                alpha_neg = numpyro.sample("alpha_neg", dist.Beta(alpha_neg_mean * alpha_neg_kappa, (1 - alpha_neg_mean) * alpha_neg_kappa))[:, None]
            else:
                alpha_neg = alpha_pos

            if model[2]:
                alpha_cf_pos = numpyro.sample("alpha_cf_pos", dist.Beta(alpha_cf_pos_mean * alpha_cf_pos_kappa, (1 - alpha_cf_pos_mean) * alpha_cf_pos_kappa))[:, None]
            else:
                alpha_cf_pos = alpha_pos

            if model[3]:
                alpha_cf_neg = numpyro.sample("alpha_cf_neg", dist.Beta(alpha_cf_neg_mean * alpha_cf_neg_kappa, (1 - alpha_cf_neg_mean) * alpha_cf_neg_kappa))[:, None]
            elif not model[3] and model[2]:
                alpha_cf_neg = alpha_cf_pos
            else:
                alpha_cf_neg = alpha_neg

            if model[4]:
                alpha_ch = numpyro.sample("alpha_ch", dist.Beta(alpha_ch_mean * alpha_ch_kappa, (1 - alpha_ch_mean) * alpha_ch_kappa))[:, None]
            else:
                alpha_ch = jnp.full((n_participants, 1), 1.0)

            if model[5]:
                beta_ch = numpyro.sample("beta_ch", dist.Beta(beta_ch_mean * beta_ch_kappa, (1 - beta_ch_mean) * beta_ch_kappa))[:, None] * beta_scaling
            else:
                beta_ch = jnp.full((n_participants, 1), 0.0)

            if model[6]:
                beta_r = numpyro.sample("beta_r", dist.Beta(beta_r_mean * beta_r_kappa, (1 - beta_r_mean) * beta_r_kappa))[:, None] * beta_scaling
            else:
                beta_r = jnp.full((n_participants, 1), 1.0)

            beta_cf = 1.0 if model[7] else 0.0
            
    else:
        # Non-hierarchical version
        alpha_pos = numpyro.sample("alpha_pos", dist.Beta(1, 1)) if model[0]==1 else 1
        alpha_neg = numpyro.sample("alpha_neg", dist.Beta(1, 1)) if model[1]==1 else alpha_pos
        alpha_cf_pos = numpyro.sample("alpha_cf_pos", dist.Beta(1, 1)) if model[2]==1 else alpha_pos
        alpha_cf_neg = numpyro.sample("alpha_cf_neg", dist.Beta(1, 1)) if model[3]==1 else alpha_neg
        alpha_ch = numpyro.sample("alpha_ch", dist.Beta(1, 1)) if model[4]==1 else 1
        beta_ch = numpyro.sample("beta_ch", scaled_beta(1, 1, 0, 15)) if model[5]==1 else 0
        beta_r = numpyro.sample("beta_r", scaled_beta(1, 1, 0, 15)) if model[6]==1 else 1
        beta_cf = 1.0 if model[7] else 0.0
    
    def rl_update(carry, x):
        """Standard RL update function"""
        r_values, c_values = carry
        ch, rw = x[:, :2], x[:, 2][:, None]
        
        # Compute prediction errors
        rpe = (rw - r_values) * ch
        rpe_cf = ((1-rw) - r_values) * (1-ch) * beta_cf
        cpe = ch - c_values
        
        # Update Q-values - vectorized operations
        if hierarchical == 1:
            lr = jnp.where(rw > 0.5, alpha_pos, alpha_neg)
            lr_cf = jnp.where(rw > 0.5, alpha_cf_pos, alpha_cf_neg)
        else:
            lr = jnp.where(rw > 0.5, alpha_pos, alpha_neg)
            lr_cf = jnp.where(rw > 0.5, alpha_cf_pos, alpha_cf_neg)
            
        r_values = r_values + lr * rpe + lr_cf * rpe_cf
        c_values = c_values + alpha_ch * cpe
        
        # Compute action probabilities
        r_diff = (r_values[:, 0] - r_values[:, 1])[:, None]
        c_diff = (c_values[:, 0] - c_values[:, 1])[:, None]
        
        if hierarchical == 1:
            action_prob_option_0 = jax.nn.sigmoid(beta_r * r_diff + beta_ch * c_diff).flatten()
        else:
            action_prob_option_0 = jax.nn.sigmoid(beta_r * r_diff + beta_ch * c_diff).flatten()
        
        return (r_values, c_values), action_prob_option_0
    
    # Initialize values
    r_values = jnp.full((n_participants, 2), 0.5)
    c_values = jnp.zeros((n_participants, 2))
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    
    # Run model forward using jax.lax.scan for efficiency
    carry_init = (r_values, c_values)
    _, ys = jax.lax.scan(rl_update, carry_init, xs)
    
    # Likelihood
    next_choice_0 = choice[1:, :, 0]
    valid_mask = (next_choice_0 >= 0) & (next_choice_0 <= 1)
    
    if hierarchical == 1:
        with numpyro.handlers.mask(mask=valid_mask):
            with numpyro.plate("participants", n_participants, dim=-1):
                with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
                    numpyro.sample("obs", dist.Bernoulli(probs=ys), obs=next_choice_0)
    else:
        with numpyro.handlers.mask(mask=valid_mask.flatten()):
            numpyro.sample("obs", dist.Bernoulli(probs=ys.flatten()), obs=next_choice_0.flatten())


def encode_model_name(model: str, model_parts: list) -> np.ndarray:
    """Encode model configuration string into binary array"""
    enc = np.zeros((len(model_parts),))
    for i in range(len(model_parts)):
        if model_parts[i] in model:
            enc[i] = 1
    return enc


def main(file: str, model: str, num_samples: int, num_warmup: int, num_chains: int, 
         hierarchical: bool, output_file: str, checkpoint: bool, train_test_ratio: float = 1., 
         d: int = 2):
    
    # Determine model type and set up accordingly
    if model.upper() == "GQL":
        model_type = "gql"
        model_function = lambda choice, reward, hierarchical: gql_model(choice, reward, hierarchical, d)
        output_file = output_file.split('.')[0] + '_GQL.nc'
        print(f"Using Generalized Q-learning model with d={d}")
    else:
        model_type = "standard"
        valid_config = ['Ap', 'An', 'Acfp', 'Acfn', 'Ach', 'Bch', 'Br', 'Bcf']
        
        # Validate model configuration
        model_checked = '' + model
        for c in valid_config:
            model_checked = model_checked.replace(c, '')
        if len(model_checked) > 0:
            raise ValueError(f'The provided model {model} is not supported. '
                           f'Invalid parts: {model_checked}. Valid parts: {valid_config}. '
                           f'Or use "GQL" for Generalized Q-learning.')
        
        model_encoded = tuple(encode_model_name(model, valid_config))
        model_function = lambda choice, reward, hierarchical: rl_model(model_encoded, choice, reward, hierarchical)
        output_file = output_file.split('.')[0] + '_' + model + '.nc'
        print(f"Using standard RL model with configuration: {model}")
    
    # Get and prepare the data
    dataset = convert_dataset(file)[0]
    if isinstance(train_test_ratio, float):
        dataset = split_data_along_timedim(dataset=dataset, split_ratio=train_test_ratio)[0].xs.numpy()
    else:
        dataset = split_data_along_sessiondim(dataset=dataset, list_test_sessions=train_test_ratio)[0].xs.numpy()
    choices = dataset[..., :2]
    rewards = np.max(dataset[..., 2:4], axis=-1, keepdims=True)
    
    # Run the model
    numpyro.set_host_device_count(num_chains)
    print(f'Number of devices: {jax.device_count()}')
    
    kernel = infer.NUTS(model_function)
    
    if checkpoint and num_warmup > 0:
        print(f'Checkpoint was set but num_warmup>0 ({num_warmup}). Setting num_warmup=0.')
        num_warmup = 0
        
    mcmc = infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    print(f'Initialized MCMC model ({model_type}).')
    
    if checkpoint:
        with open(output_file, 'rb') as file:
            checkpoint_data = pickle.load(file)
        mcmc.post_warmup_state = checkpoint_data.last_state
        rng_key = mcmc.post_warmup_state.rng_key
        print('Checkpoint loaded.')
    else:
        rng_key = jax.random.PRNGKey(0)
        
    mcmc.run(rng_key, 
             choice=jnp.array(choices.swapaxes(1, 0)), 
             reward=jnp.array(rewards.swapaxes(1, 0)), 
             hierarchical=hierarchical)

    with open(output_file, 'wb') as file:
        pickle.dump(mcmc, file)
    
    return mcmc


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Hierarchical Bayesian parameter inference with NumPyro.')
  
    parser.add_argument('--file', type=str, help='Dataset of a 2-armed bandit task')
    parser.add_argument('--model', type=str, default='ApAnAchBchBr', 
                       help='Model configuration (e.g., "ApAnAchBchBr" for standard RL, "GQL" for Generalized Q-learning)')
    parser.add_argument('--d', type=int, default=2, help='Number of Q-value/history components for GQL model (default=2 as in paper)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=500, help='Number of warmup samples')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    parser.add_argument('--hierarchical', action='store_true', help='Whether to do hierarchical inference')
    parser.add_argument('--output_file', type=str, default='benchmarking/params/traces.nc', help='Output file')
    parser.add_argument('--checkpoint', action='store_true', help='Whether to load checkpoint')
    
    args = parser.parse_args()

    main(args.file, args.model, args.num_samples, args.num_warmup, args.num_chains, 
         args.hierarchical, args.output_file, args.checkpoint, d=args.d)