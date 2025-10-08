"""
Comprehensive Feature/Behavioral Analysis Script
Creates seaborn feature plots from participant_features.csv across diagnoses
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Change to the script directory
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# Set pandas option
pd.set_option('future.no_silent_downcasting', True)

# Load the participant features data
features_path = '../features/participant_features.csv'
df = pd.read_csv(features_path)
print(f"Loaded data from {features_path}")
print(f"Data shape: {df.shape}")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())


