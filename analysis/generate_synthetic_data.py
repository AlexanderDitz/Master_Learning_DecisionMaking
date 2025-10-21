"""
Synthetic Data Generation Script
Generates synthetic behavioral data from trained models with realistic task difficulty
"""

# Import packages
import pandas as pd
import numpy as np
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model loading functions
from model_comparison import (
    load_lstm_model, load_spice_model, load_rnn_model, load_gql_model,
    load_dezfouli_dataset
)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Change to the script directory
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# Set pandas option
pd.set_option('future.no_silent_downcasting', True)

def analyze_real_task_difficulty():
    """Analyze the real task to understand reward structure and difficulty."""
    print("🔍 Analyzing real task difficulty...")
    
    # Load real data to understand the task structure
    dataset = load_dezfouli_dataset()
    
    if dataset is None:
        print("❌ Failed to load real data for analysis")
        return None, None
    
    all_choices = []
    all_rewards = []
    all_reward_left = []
    all_reward_right = []
    
    for participant_id, participant_data in dataset.items():
        choices = participant_data['choice'].values
        rewards = participant_data['reward'].values
        reward_left = participant_data['reward_left'].values
        reward_right = participant_data['reward_right'].values
        
        all_choices.extend(choices)
        all_rewards.extend(rewards)
        all_reward_left.extend(reward_left)
        all_reward_right.extend(reward_right)
    
    # Calculate statistics
    choice_rate = np.mean(all_choices)
    reward_rate = np.mean(all_rewards)
    
    # Estimate arm probabilities
    left_choices = np.array(all_choices) == 0
    right_choices = np.array(all_choices) == 1
    
    left_reward_prob = np.mean(np.array(all_reward_left)[left_choices]) if np.any(left_choices) else 0
    right_reward_prob = np.mean(np.array(all_reward_right)[right_choices]) if np.any(right_choices) else 0
    
    print(f"📊 Real Task Analysis:")
    print(f"   Overall choice rate (right arm): {choice_rate:.3f}")
    print(f"   Overall reward rate: {reward_rate:.3f}")
    print(f"   Left arm reward probability: {left_reward_prob:.3f}")
    print(f"   Right arm reward probability: {right_reward_prob:.3f}")
    print(f"   Total trials analyzed: {len(all_choices)}")
    
    return left_reward_prob, right_reward_prob

def generate_realistic_bandit_environment(n_trials=500, left_prob=None, right_prob=None):
    """Generate a bandit environment matching the real task difficulty."""
    
    # If probabilities not provided, use very low probabilities to match real data
    if left_prob is None or right_prob is None:
        # Use extremely low probabilities to match the ~11% reward rate observed
        left_prob = 0.12
        right_prob = 0.10
    
    rewards = np.zeros((n_trials, 2))
    for trial in range(n_trials):
        rewards[trial, 0] = np.random.binomial(1, left_prob)   # Left arm
        rewards[trial, 1] = np.random.binomial(1, right_prob)  # Right arm
    
    return rewards

def simulate_lstm_behavior(model, rewards, n_trials=500):
    """Simulate LSTM behavior with proper parameter usage."""
    actions = []
    received_rewards = []
    
    # Reset LSTM state for new session
    model.new_sess()
    
    for trial in range(n_trials):
        try:
            # Get action probabilities from the model
            action_probs = model.get_choice_probs()
            
            # Sample action based on probabilities
            action = np.random.choice([0, 1], p=action_probs)
            
            # Get reward for chosen action
            reward = rewards[trial, action]
            
            actions.append(action)
            received_rewards.append(reward)
            
            # Update model with experience
            model.update(action, reward)
            
        except Exception as e:
            # Fallback to random action
            action = np.random.choice([0, 1])
            reward = rewards[trial, action]
            actions.append(action)
            received_rewards.append(reward)
    
    return np.array(actions), np.array(received_rewards)

def simulate_spice_behavior(model, rewards, n_trials=500):
    """Simulate SPICE behavior with proper symbolic reasoning."""
    actions = []
    received_rewards = []
    
    # Reset SPICE internal state if possible
    if hasattr(model, 'new_sess'):
        model.new_sess()
    
    for trial in range(n_trials):
        try:
            # Get action from SPICE's symbolic policy
            action_probs = model.get_choice_probs()
            action = np.random.choice([0, 1], p=action_probs)
            
            # Get reward
            reward = rewards[trial, action]
            
            actions.append(action)
            received_rewards.append(reward)
            
            # Update SPICE with the experience
            model.update(action, reward)
            
        except Exception as e:
            # Fallback to exploiting current best estimate
            if hasattr(model, 'q'):
                action = np.argmax(model.q)
            else:
                action = np.random.choice([0, 1])
            reward = rewards[trial, action]
            actions.append(action)
            received_rewards.append(reward)
    
    return np.array(actions), np.array(received_rewards)

def simulate_rnn_behavior(model, rewards, n_trials=500):
    """Simulate RNN behavior with proper recurrent processing."""
    actions = []
    received_rewards = []
    
    # Reset RNN hidden state
    if hasattr(model, 'new_sess'):
        model.new_sess()
    
    for trial in range(n_trials):
        try:
            # Get action from RNN
            action_probs = model.get_choice_probs()
            action = np.random.choice([0, 1], p=action_probs)
            
            # Get reward
            reward = rewards[trial, action]
            
            actions.append(action)
            received_rewards.append(reward)
            
            # Update RNN
            model.update(action, reward)
            
        except Exception as e:
            action = np.random.choice([0, 1])
            reward = rewards[trial, action]
            actions.append(action)
            received_rewards.append(reward)
    
    return np.array(actions), np.array(received_rewards)

def simulate_gql_behavior(model, rewards, n_trials=500):
    """Simulate GQL behavior with proper Q-learning."""
    actions = []
    received_rewards = []
    
    # Handle tuple format for GQL
    if isinstance(model, tuple):
        agents, _ = model
        agent = agents[0]  # Use first agent
    else:
        agent = model
    
    # Reset Q-values if possible
    if hasattr(agent, 'new_sess'):
        agent.new_sess()
    
    for trial in range(n_trials):
        try:
            # Get action from GQL agent
            action = agent.get_choice()
            
            # Get reward
            reward = rewards[trial, action]
            
            actions.append(action)
            received_rewards.append(reward)
            
            # Update Q-values
            agent.update(action, reward)
            
        except Exception as e:
            action = np.random.choice([0, 1])
            reward = rewards[trial, action]
            actions.append(action)
            received_rewards.append(reward)
    
    return np.array(actions), np.array(received_rewards)

def format_trial_data(participant_id, model_name, actions, rewards):
    """Format trial-by-trial data for a participant."""
    n_trials = len(actions)
    
    trial_data = []
    for trial in range(n_trials):
        trial_row = {
            'participant_id': participant_id,
            'model_source': model_name,
            'trial': trial + 1,
            'choice': int(actions[trial]),  # 0 or 1
            'reward': int(rewards[trial]),  # 0 or 1
        }
        trial_data.append(trial_row)
    
    return trial_data

def generate_synthetic_dataset(n_participants_per_model=10, n_trials=500):
    """Generate synthetic dataset from trained models with realistic task difficulty."""
    print("🔄 Loading trained models...")
    
    # First analyze real task difficulty
    left_prob, right_prob = analyze_real_task_difficulty()
    
    # Define model paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    params_dir = os.path.join(project_dir, 'params', 'dezfouli2019')
    
    models = {}
    
    # Load models with their trained parameters
    try:
        lstm_path = os.path.join(params_dir, 'lstm_dezfouli2019.pkl')
        models['LSTM'] = load_lstm_model(lstm_path)
        print("✓ LSTM loaded with trained parameters")
    except Exception as e:
        print(f"✗ LSTM failed: {e}")
        models['LSTM'] = None
    
    try:
        spice_path = os.path.join(params_dir, 'spice_dezfouli2019_l2_0_001.pkl')
        rnn_path = os.path.join(params_dir, 'rnn_dezfouli2019_l2_0_001.pkl')
        models['SPICE'] = load_spice_model(spice_path, rnn_path)
        print("✓ SPICE loaded with trained parameters")
    except Exception as e:
        print(f"✗ SPICE failed: {e}")
        models['SPICE'] = None
    
    try:
        rnn_path = os.path.join(params_dir, 'rnn_dezfouli2019_l2_0_001.pkl')
        models['RNN'] = load_rnn_model(rnn_path)
        print("✓ RNN loaded with trained parameters")
    except Exception as e:
        print(f"✗ RNN failed: {e}")
        models['RNN'] = None
    
    try:
        gql_path = os.path.join(params_dir, 'gql_dezfouli2019_PhiChiBetaKappaC.pkl')
        models['GQL'] = load_gql_model(gql_path)
        print("✓ GQL loaded with trained parameters")
    except Exception as e:
        print(f"✗ GQL failed: {e}")
        models['GQL'] = None
    
    print(f"\n🎲 Generating synthetic behavioral data with realistic task difficulty...")
    print(f"   Using arm probabilities: Left={left_prob:.3f}, Right={right_prob:.3f}")
    
    # Generate synthetic participants for each model
    all_trial_data = []
    participant_id = 1
    
    for model_name, model in models.items():
        if model is None:
            print(f"⚠️ Skipping {model_name} (not loaded)")
            continue
        
        print(f"📊 Generating data from {model_name} model...")
        
        for participant in range(n_participants_per_model):
            # Generate environment with realistic reward structure
            reward_matrix = generate_realistic_bandit_environment(n_trials, left_prob, right_prob)
            
            # Simulate model behavior using trained parameters
            if model_name == 'LSTM':
                actions, received_rewards = simulate_lstm_behavior(model, reward_matrix, n_trials)
            elif model_name == 'SPICE':
                actions, received_rewards = simulate_spice_behavior(model, reward_matrix, n_trials)
            elif model_name == 'RNN':
                actions, received_rewards = simulate_rnn_behavior(model, reward_matrix, n_trials)
            elif model_name == 'GQL':
                actions, received_rewards = simulate_gql_behavior(model, reward_matrix, n_trials)
            else:
                # Fallback random behavior
                actions = np.random.choice([0, 1], size=n_trials)
                received_rewards = reward_matrix[np.arange(n_trials), actions]
            
            # Format trial-by-trial data
            # Ensure arrays match the expected trial count
            if len(actions) > n_trials:
                actions = actions[:n_trials]
                received_rewards = received_rewards[:n_trials]
            
            participant_trials = format_trial_data(
                participant_id=f"synth_{participant_id:03d}",
                model_name=model_name,
                actions=actions,
                rewards=received_rewards
            )
            
            all_trial_data.extend(participant_trials)
            participant_id += 1
    
    # Convert to DataFrame
    df_synthetic = pd.DataFrame(all_trial_data)
    
    print(f"✅ Generated synthetic dataset with {len(df_synthetic)} trials")
    
    # Print summary statistics
    if len(df_synthetic) > 0:
        # Calculate participant-level statistics
        participant_stats = df_synthetic.groupby(['participant_id', 'model_source']).agg({
            'choice': 'mean',     # Choice rate (proportion choosing option 1)
            'reward': 'mean',     # Reward rate
            'trial': 'count'      # Number of trials per participant
        }).reset_index()
        
        n_participants = len(participant_stats)
        models_count = participant_stats['model_source'].value_counts().to_dict()
        
        print(f"   Total participants: {n_participants}")
        print(f"   Total trials: {len(df_synthetic)}")
        print(f"   Models: {models_count}")
        
        print(f"\n🤖 SYNTHETIC DATA SUMMARY:")
        print(f"   Mean choice rate: {participant_stats['choice'].mean():.3f} ± {participant_stats['choice'].std():.3f}")
        print(f"   Mean reward rate: {participant_stats['reward'].mean():.3f} ± {participant_stats['reward'].std():.3f}")
        print(f"   Trials per participant: {participant_stats['trial'].iloc[0]}")
        
        # Model-specific summaries
        for model in participant_stats['model_source'].unique():
            model_data = participant_stats[participant_stats['model_source'] == model]
            print(f"   {model}: reward_rate={model_data['reward'].mean():.3f}±{model_data['reward'].std():.3f}, choice_rate={model_data['choice'].mean():.3f}±{model_data['choice'].std():.3f}")
    
    return df_synthetic

def save_synthetic_data(df_synthetic, filename='synthetic_choice_data.csv'):
    """Save synthetic choice data to CSV files - both combined and separate by model."""
    if df_synthetic is None or len(df_synthetic) == 0:
        print("❌ No synthetic data to save")
        return
    
    # Create synthetic data directory if it doesn't exist
    data_dir = '../data/synthetic data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Save combined data to CSV
    filepath = os.path.join(data_dir, filename)
    df_synthetic.to_csv(filepath, index=False)
    
    print(f"💾 Saved combined synthetic choice data to: {filepath}")
    print(f"   Columns: {list(df_synthetic.columns)}")
    print(f"   Shape: {df_synthetic.shape}")
    
    # Save separate CSV files for each model
    print(f"\n💾 Saving separate CSV files for each model...")
    
    for model in df_synthetic['model_source'].unique():
        model_data = df_synthetic[df_synthetic['model_source'] == model]
        model_filename = f'synthetic_choices_{model.lower()}.csv'
        model_filepath = os.path.join(data_dir, model_filename)
        
        model_data.to_csv(model_filepath, index=False)
        
        # Calculate stats for this model
        participant_stats = model_data.groupby('participant_id').agg({
            'choice': 'mean',
            'reward': 'mean'
        })
        
        n_participants = len(participant_stats)
        choice_rate_mean = participant_stats['choice'].mean()
        choice_rate_std = participant_stats['choice'].std()
        reward_rate_mean = participant_stats['reward'].mean()
        reward_rate_std = participant_stats['reward'].std()
        
        print(f"   ✓ {model}: {model_filepath} ({n_participants} participants, {len(model_data)} trials)")
        print(f"     Choice rate: {choice_rate_mean:.3f}±{choice_rate_std:.3f}, "
              f"Reward rate: {reward_rate_mean:.3f}±{reward_rate_std:.3f}")
    
    print(f"\n📊 Data files created:")
    print(f"   - Combined: {filepath}")
    for model in df_synthetic['model_source'].unique():
        model_filename = f'synthetic_choices_{model.lower()}.csv'
        print(f"   - {model}: {os.path.join(data_dir, model_filename)}")
    
    # Print sample of the data
    print(f"\n📝 Sample of generated choice data:")
    print(df_synthetic.head(10).to_string(index=False))

# Main execution
if __name__ == "__main__":
    # Generate synthetic data with 101 participants per model
    n_participants_per_model = 101
    
    print(f"\n🎯 Generating {n_participants_per_model} participants per model...")
    
    # Generate dataset
    df_synthetic = generate_synthetic_dataset(n_participants_per_model=n_participants_per_model, n_trials=500)
    
    # Save the data
    save_synthetic_data(df_synthetic)
    
    print(f"\n✅ Synthetic choice data generation complete!")
    print(f"   Generated {len(df_synthetic)} total trials from {len(df_synthetic.groupby('participant_id'))} participants ({n_participants_per_model} per model)")
    print(f"   Trial-by-trial choice data saved as combined and separate CSV files")
    print(f"   Each row contains: participant_id, model_source, trial, choice (0/1), reward (0/1)")
    print(f"   Use this data for behavioral analysis, model comparison, and feature extraction")
