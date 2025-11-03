#!/usr/bin/env python3
"""
Extract SINDy (cognitive) equations from SPICE model fit to Dezfouli 2019 dataset.
Outputs a CSV file with coefficients for each participant and module.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import pandas as pd
from utils.model_loading_utils import load_dezfouli_dataset, load_spice_model

# Model and data paths
base_params = "params/dezfouli2019"
spice_path = os.path.join(base_params, "spice2_dezfouli2019_l2_0_001.pkl")
rnn_path = os.path.join(base_params, "rnn_dezfouli2019_l2_0_001.pkl")

print("📂 Loading Dezfouli 2019 dataset...")
dataset = load_dezfouli_dataset()
if dataset is None:
    print("❌ Failed to load dataset. Exiting.")
    sys.exit(1)
print(f"✓ Loaded dataset: {len(dataset)} participants")

print("\n🤖 Loading SPICE model...")
agent_spice = load_spice_model(spice_path, rnn_path, deterministic=False)
if agent_spice is None:
    print("❌ Failed to load SPICE model. Exiting.")
    sys.exit(1)
print("✓ SPICE model loaded!")

import csv

# Step 1: Collect all module+coefficient column names
all_columns = set()
participant_rows = []
for idx, (pid, pdata) in enumerate(dataset.items()):
    participant_id = idx
    agent_spice.new_sess(participant_id=participant_id)
    modules = agent_spice.get_modules()
    row = {'participant': pid}
    for module in modules:
        available_keys = list(modules[module].keys())
        coeff = None
        for k in available_keys:
            if int(k) == participant_id or (hasattr(k, 'item') and int(k.item()) == participant_id):
                coeff = modules[module][k].coefficients()
                break
        if coeff is not None:
            if hasattr(coeff, 'tolist'):
                coeff = coeff.tolist()
            if isinstance(coeff, (list, tuple, np.ndarray)):
                for i, val in enumerate(coeff):
                    colname = f"{module}_{i}"
                    row[colname] = val
                    all_columns.add(colname)
            else:
                colname = f"{module}_0"
                row[colname] = coeff
                all_columns.add(colname)
        else:
            print(f"Participant {pid}, Module {module}: not found. Available keys: {[f'{k} (type: {type(k)})' for k in available_keys]}")
    participant_rows.append(row)

# Step 2: Build DataFrame with all columns
all_columns = sorted(all_columns)
df = pd.DataFrame(participant_rows)
df = df.set_index('participant')
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan
# Reorder columns
cols = ['participant'] + all_columns if 'participant' in df.columns else all_columns
df = df[all_columns]

# Step 3: Save wide-format CSV for clustering
out_path = "analysis/spice_sindy_parameters.csv"
df.to_csv(out_path)
print(f"✅ Saved wide-format SINDy coefficients for clustering to {out_path}")