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
    'win_stay', 'win_shift', 'lose_stay', 'lose_shift'
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
synthetic_data_dir = '../synthetic data'
synthetic_files = {
    'LSTM': 'synthetic_choices_lstm.csv',
    'SPICE': 'synthetic_choices_spice.csv', 
    'RNN': 'synthetic_choices_rnn.csv',
    'GQL': 'synthetic_choices_gql.csv'
}

# Check if synthetic data exists
if not os.path.exists(synthetic_data_dir):
    print(f"❌ Synthetic data directory not found: {synthetic_data_dir}")
    print("   Run generate_synthetic_data.py first to create synthetic data")
else:
    all_synthetic_rows = []
    
    for model_name, filename in synthetic_files.items():
        filepath = os.path.join(synthetic_data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"📊 Processing {model_name} data from {filename}...")
            
            # Load synthetic data
            synthetic_df = pd.read_csv(filepath)
            print(f"   Loaded {len(synthetic_df)} trials from {len(synthetic_df['participant_id'].unique())} participants")
            
            # Rename columns to match the format expected by compute_participant_features
            synthetic_df_renamed = synthetic_df.rename(columns={
                'participant_id': 'df_participant_id',
                'choice': 'df_choice', 
                'reward': 'df_reward'
            })
            
            # Process each synthetic participant
            for pid, group in synthetic_df_renamed.groupby('df_participant_id'):
                feats = compute_participant_features(group)
                feats["participant"] = pid
                feats["diagnosis"] = model_name  # Use model name as "diagnosis"
                all_synthetic_rows.append(feats)
            
            print(f"   ✓ Processed {len(synthetic_df['participant_id'].unique())} {model_name} participants")
        else:
            print(f"⚠️ Synthetic file not found: {filepath}")
    
    if all_synthetic_rows:
        # Create DataFrame of synthetic participant features
        synthetic_participant_df = pd.DataFrame(all_synthetic_rows)
        
        # Rearrange columns to match real data format
        synthetic_participant_df = synthetic_participant_df[desired_order]
        
        # Create synthetic_features subfolder and save synthetic features
        os.makedirs('../features/synthetic_features', exist_ok=True)
        synthetic_features_path = '../features/synthetic_features/synthetic_participant_features.csv'
        synthetic_participant_df.to_csv(synthetic_features_path, index=False)
        
        print(f"\n💾 Saved synthetic participant features to '{synthetic_features_path}'")
        print("Sample synthetic features:")
        print(synthetic_participant_df.head())
        print("\nModel distribution in synthetic features:")
        print(synthetic_participant_df['diagnosis'].value_counts())
        print(f"✅ Created synthetic features for {len(synthetic_participant_df)} participants")
        
        # Display comparison statistics
        print("\n📊 COMPARISON SUMMARY:")
        print(f"Real participants: {len(participant_df)}")
        print(f"Synthetic participants: {len(synthetic_participant_df)}")
        
        print("\nReal data diagnosis distribution:")
        for diag, count in participant_df['diagnosis'].value_counts().items():
            print(f"  {diag}: {count}")
            
        print("\nSynthetic data model distribution:")
        for model, count in synthetic_participant_df['diagnosis'].value_counts().items():
            print(f"  {model}: {count}")
    else:
        print("❌ No synthetic data files found to process")