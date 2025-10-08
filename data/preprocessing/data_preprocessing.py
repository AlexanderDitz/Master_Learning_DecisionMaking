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
    choices = group['choice'].values
    rewards = group['reward'].values

    n_trials = len(choices)
    choice_rate = choices.mean()
    reward_rate = rewards.mean()

    win_stay, win_shift, lose_stay, lose_shift = [], [], [], []

    for t in range(1, n_trials):
        prev_choice = choices[t - 1]
        curr_choice = choices[t]
        prev_reward = rewards[t - 1]

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
        "n_trials": n_trials
    }
    return features

