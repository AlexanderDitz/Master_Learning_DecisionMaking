# Import packages
import numpy as np
import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Change to the script directory
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

pd.set_option('future.no_silent_downcasting', True)

# Load the data file (csv-format) - now using relative path from script location
df = pd.read_csv('original_data.csv')

# Visualizing the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Summary statistics
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Change key to integer instead of object
# Define the mapping
key_mapping = {'R1': 0, 'R2': 1}
# Apply the mapping
df['key'] = df['key'].replace(key_mapping)

# Rename the columns correctly
df = df.rename(columns={'ID': 'df_participant_id', 'block': 'df_session', 'key': 'df_choice', 'reward': 'df_reward'})

# Convert "choice" and "reward" column to float
df = df.astype({'df_choice': float, 'df_reward': float})

# Drop unneeded columns "diag" and "best_action"
df = df.drop(columns=['diag', 'best_action'])

# Rearrange columns in a specific order
df = df[['df_participant_id', 'df_session', 'df_choice', 'df_reward']]

# Export to CSV
df.to_csv('dezfouli2019.csv', index=False)
print(f"\nProcessed data saved to 'dezfouli2019.csv'")
print(f"Final dataset shape: {df.shape}")

# Load the cleaned dataset
df = pd.read_csv('dezfouli2019.csv')

# Load original data to get diagnosis information
original_df = pd.read_csv('original_data.csv')

# Create participant-to-diagnosis mapping
participant_diagnosis_map = original_df.groupby('ID')['diag'].first().to_dict()

print(f"Created diagnosis mapping for {len(participant_diagnosis_map)} participants")
print("Sample mappings:", {k: v for i, (k, v) in enumerate(participant_diagnosis_map.items()) if i < 3})

# === Define feature computation per participant ===
def compute_participant_features(group):
    choices = group['df_choice'].values
    rewards = group['df_reward'].values

    n_trials = len(choices)
    choice_rate = choices.mean()
    reward_rate = rewards.mean()

    win_stay, win_shift, lose_stay, lose_shift, choice_perseveration, switch_rate = [], [], [], [], [], []

    for t in range(1, n_trials):
        prev_choice = choices[t - 1]
        curr_choice = choices[t]
        prev_reward = rewards[t - 1]

        # Choice perseveration: Tendency to repeat previous choice
        choice_perseveration.append(curr_choice == prev_choice)
        # Switch rate: Tendency to switch in the next trial
        switch_rate.append(curr_choice != prev_choice)

        if prev_reward == 1:
            win_stay.append(curr_choice == prev_choice)
            win_shift.append(curr_choice != prev_choice)
        else:
            lose_stay.append(curr_choice == prev_choice)
            lose_shift.append(curr_choice != prev_choice)

    features = {
        "choice_rate": choice_rate,
        "reward_rate": reward_rate,
        "win_stay": np.mean(win_stay) if win_stay else np.nan,
        "win_shift": np.mean(win_shift) if win_shift else np.nan,
        "lose_stay": np.mean(lose_stay) if lose_stay else np.nan,
        "lose_shift": np.mean(lose_shift) if lose_shift else np.nan,
        "choice_perseveration": np.mean(choice_perseveration) if choice_perseveration else np.nan,
        "switch_rate": np.mean(switch_rate) if switch_rate else np.nan,
        "n_trials": n_trials
    }
    return features

# === Group by participant only, compute participant-level features ===
participant_rows = []

for pid, group in df.groupby('df_participant_id'):  # Use the correct column name
    feats = compute_participant_features(group)
    feats["participant"] = pid
    # Add diagnosis information
    diagnosis = participant_diagnosis_map.get(pid, "Unknown")
    feats["diagnosis"] = diagnosis
    participant_rows.append(feats)

# Create DataFrame of participant-level features
participant_df = pd.DataFrame(participant_rows)

# Rearranging columns before saving
desired_order = [
    'participant', 'diagnosis', 'n_trials',
    'choice_rate', 'reward_rate',
    'win_stay', 'win_shift', 'lose_stay', 'lose_shift', 
    'choice_perseveration', 'switch_rate'
]

participant_df = participant_df[desired_order]

# Create the features directory and subfolders if they don't exist
os.makedirs('../features/real_features', exist_ok=True)

# Save to CSV in the real_features subfolder
participant_df.to_csv('../features/real_features/real_participant_features.csv', index=False)

print("Saved participant-level features to '../features/real_features/real_participant_features.csv'")
print(participant_df.head())
print("\nDiagnosis distribution in generated features:")
print(participant_df['diagnosis'].value_counts())

print(f"✅ Created features for {len(participant_df)} participants")

# === Process Synthetic Data ===
print("\n" + "="*60)
print("PROCESSING SYNTHETIC DATA")
print("="*60)

# Define paths to synthetic data files
synthetic_data_dir = '../synthetic_data'

# Uncomment the file you want to inspect
# filename = 'dezfouli2019_generated_behavior_lstm.csv'
# filename = 'dezfouli2019_generated_behavior_spice2_l2_0_001.csv'
# filename = 'dezfouli2019_generated_behavior_spice3_l2_0_0001.csv'
# filename = 'dezfouli2019_generated_behavior_spice4_l2_0_00001.csv'
# filename = 'dezfouli2019_generated_behavior_spice5_l2_0_0005.csv'
# filename = 'dezfouli2019_generated_behavior_spice6_l2_0_00005.csv'
# filename = 'dezfouli2019_generated_behavior_rnn_l2_0_001.csv'
# filename = 'dezfouli2019_generated_behavior_rnn2_l2_0_0001.csv'
# filename = 'dezfouli2019_generated_behavior_rnn3_l2_0_00001.csv'
# filename = 'dezfouli2019_generated_behavior_rnn4_l2_0_0005.csv'
# filename = 'dezfouli2019_generated_behavior_rnn5_l2_0_00005.csv'
# filename = 'dezfouli2019_generated_behavior_benchmark.csv'
filename = 'dezfouli2019_generated_behavior_q_agent.csv'

synthetic_df = pd.read_csv(os.path.join(synthetic_data_dir, filename))
print(f"Loaded synthetic data from {filename} with shape {synthetic_df.shape}")

# --- Harmonize column names to match real data conventions ---
synthetic_df = synthetic_df.rename(columns={
    'id': 'df_participant_id',
    'session': 'df_session',
    'choice': 'df_choice',
    'reward': 'df_reward'
})

# Make sure numeric columns are correctly typed
synthetic_df = synthetic_df.astype({
    'df_session': int,
    'df_choice': float,
    'df_reward': float
})

print("Synthetic dataset columns after renaming:", synthetic_df.columns.tolist())
print("First few rows:")
print(synthetic_df.head())

# === Compute participant-level features for synthetic data ===
synthetic_participant_rows = []

for pid, group in synthetic_df.groupby('df_participant_id'):
    feats = compute_participant_features(group)
    feats["participant"] = pid
    feats["model_type"] = group["model_type"].iloc[0] if "model_type" in group.columns else "unknown"
    synthetic_participant_rows.append(feats)

synthetic_participant_df = pd.DataFrame(synthetic_participant_rows)

# Reorder columns for consistency
desired_order_synth = [
    'participant', 'model_type', 'n_trials',
    'choice_rate', 'reward_rate',
    'win_stay', 'win_shift', 'lose_stay', 'lose_shift',
    'choice_perseveration', 'switch_rate'
]
synthetic_participant_df = synthetic_participant_df[desired_order_synth]

# Create output folder
os.makedirs('../features/synthetic_features', exist_ok=True)

# Save to CSV
synthetic_features_path = f'../features/synthetic_features/synthetic_features_{os.path.splitext(filename)[0]}.csv'
synthetic_participant_df.to_csv(synthetic_features_path, index=False)

print(f"✅ Saved synthetic participant-level features to '{synthetic_features_path}'")
print(synthetic_participant_df.head())

# --- Optional: quick comparison summary ---
print("\nSummary comparison:")
print(f"Real data participants: {len(participant_df)}")
print(f"Synthetic data participants: {len(synthetic_participant_df)}")

# Compute simple mean comparison
feature_cols = ['choice_rate', 'reward_rate', 'win_stay', 'win_shift', 'lose_stay', 'lose_shift', 'choice_perseveration', 'switch_rate']
comparison_df = pd.DataFrame({
    'Real (mean)': participant_df[feature_cols].mean(),
    'Synthetic (mean)': synthetic_participant_df[feature_cols].mean()
})
print("\nAverage feature comparison (real vs synthetic):")
print(comparison_df)