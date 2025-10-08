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

