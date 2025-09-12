import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from resources.bandits import AgentQ, get_update_dynamics
from resources.rnn_utils import split_data_along_sessiondim
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.plotting import plot_session
from utils.convert_dataset import convert_dataset
from resources import rnn, sindy_utils, rnn_utils
from resources.model_evaluation import log_likelihood

from benchmarking.benchmarking_lstm import setup_agent_lstm
from benchmarking import benchmarking_eckstein2022, benchmarking_dezfouli2019

# ----------------------------------------------------------------------------------------------
# Eckstein 2022
# ----------------------------------------------------------------------------------------------

# study = 'eckstein2022'

# L_baseline, NLL_baseline, AIC_baseline, BIC_baseline = 0.67622, 0.39124, 0.81364, 0.86432
# L_benchmark, NLL_benchmark, AIC_benchmark, BIC_benchmark = 0.70341, 0.35182, 0.74656, 0.84791
# L_LSTM, NLL_LSTM, AIC_LSTM, BIC_LSTM = 0.69989, 0.35684, 28.30485, 64.84373

# # L̄ (L_bar) values
# L_RNN = [0.69346, 0.69620, 0.70280, 0.70560, 0.70459, 0.69702]
# L_SPICE = [0.61740, 0.68266, 0.70148, 0.70348, 0.70288, 0.69577]

# # NLL (Negative Log-Likelihood) values
# NLL_RNN = [0.36606, 0.36211, 0.35269, 0.34870, 0.35014, 0.36093]
# NLL_SPICE = [0.48224, 0.38176, 0.35457, 0.35172, 0.35257, 0.36273]

# # AIC (Akaike Information Criterion) values
# AIC_RNN = [11.10921, 11.07344, 11.09775, 11.09763, 11.15791, 11.18507]
# AIC_SPICE = [1.09541, 0.95644, 0.93909, 0.93633, 0.94384, 0.94101]

# # BIC (Bayesian Information Criterion) values
# BIC_RNN = [24.96966, 24.93389, 24.95819, 24.95808, 25.01836, 25.04551]
# BIC_SPICE = [1.43259, 1.32211, 1.32360, 1.31191, 1.26878, 1.23005]

# # L2 regularization values (for reference)
# l2_values = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]


# ----------------------------------------------------------------------------------------------
# Dezfouli 2019
# ----------------------------------------------------------------------------------------------

study = 'dezfouli2019'

# Baseline, Benchmark, LSTM values (using test data for L and NLL, train data for AIC/BIC)
L_baseline, NLL_baseline, AIC_baseline, BIC_baseline = 0.56217, 0.57595, 1.17127, 1.21979
L_benchmark, NLL_benchmark, AIC_benchmark, BIC_benchmark = 0.71490, 0.33561, 0.88018, 1.17128
L_LSTM, NLL_LSTM, AIC_LSTM, BIC_LSTM = 0.75937, 0.27526, 91.95339, 211.54895

# Training L̄ values
# L_train_RNN = [0.73883, 0.75149, 0.75735, 0.75052, 0.71755, 0.71586]
# L_train_SPICE = [0.68449, 0.65993, 0.62639, 0.54712, 0.71736, 0.71410]

# Test L̄ values  
L_RNN = [0.73610, 0.74707, 0.75640, 0.74964, 0.71958, 0.71773]
L_SPICE = [0.67465, 0.62257, 0.58965, 0.54715, 0.71943, 0.71656]

# NLL (Negative Log-Likelihood) values (from test data)
NLL_RNN = [0.30639, 0.29159, 0.27919, 0.28816, 0.32909, 0.33166]
NLL_SPICE = [0.39355, 0.47391, 0.52823, 0.60302, 0.32930, 0.33329]

# AIC (Akaike Information Criterion) values (from train data)
AIC_RNN = [10.74813, 10.71415, 10.69861, 10.71672, 10.80658, 10.81131]
AIC_SPICE = [0.99608, 1.17383, 1.24381, 1.47696, 0.82499, 0.82255]

# BIC (Bayesian Information Criterion) values (from train data)
BIC_RNN = [24.01766, 23.98367, 23.96814, 23.98625, 24.07611, 24.08083]
BIC_SPICE = [1.30331, 1.61884, 1.64611, 1.82962, 1.03512, 1.01700]

# L2 regularization values (for reference)
l2_values = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]


# ----------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------


x_values = np.arange(0, len(l2_values))

# Colors
orange = '#FF8C00'
pink = '#FF69B4'
light_grey = '#B0B0B0'  # Baseline
dark_grey = '#2F2F2F'   # Benchmark  
black = '#000000'       # LSTM

# Plot 1: L̄ values
fig1, ax = plt.subplots(figsize=(6, 5))
ax.plot(x_values, L_RNN, 'x-', color=orange, linestyle='--', label='RNN', markersize=8, linewidth=2)
ax.plot(x_values, L_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=L_baseline, color=light_grey, linestyle=':', label='Baseline', linewidth=1.5)
ax.axhline(y=L_benchmark, color=dark_grey, linestyle='--', label='Benchmark', linewidth=1.5)
ax.axhline(y=L_LSTM, color=black, linestyle='-.', label='LSTM', linewidth=1.5)
# ax.set_xlabel('L2 Regularization')
# ax.set_ylabel(r'$\bar{\mathcal{L}}$')
# ax.set_xscale('log')
# ax.grid(True, alpha=0.3)
ax.set_xticklabels(['']*len(x_values))
# ax.legend()
plt.tight_layout()
plt.savefig(f'analysis/plots_model_evaluation/{study}_L_bar.png', dpi=500)
plt.show()

# Plot 2: NLL values
fig2, ax = plt.subplots(figsize=(6, 5))
ax.plot(x_values, NLL_RNN, 'x-', color=orange, linestyle='--', label='RNN', markersize=8, linewidth=2)
ax.plot(x_values, NLL_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=NLL_baseline, color=light_grey, linestyle=':', label='Baseline', linewidth=1.5)
ax.axhline(y=NLL_benchmark, color=dark_grey, linestyle='--', label='Benchmark', linewidth=1.5)
ax.axhline(y=NLL_LSTM, color=black, linestyle='-.', label='LSTM', linewidth=1.5)
# ax.set_xlabel('L2 Regularization')
# ax.set_ylabel('NLL')
# ax.set_xscale('log')
# ax.grid(True, alpha=0.3)
# ax.legend()
ax.set_xticklabels(['']*len(x_values))
plt.tight_layout()
plt.savefig(f'analysis/plots_model_evaluation/{study}_NLL.png', dpi=500)
plt.show()

# Plot 3: AIC values (excluding RNN and LSTM baselines due to high values)
fig3, ax = plt.subplots(figsize=(6, 5))
ax.plot(x_values, AIC_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=AIC_baseline, color=light_grey, linestyle=':', label='Baseline', linewidth=1.5)
ax.axhline(y=AIC_benchmark, color=dark_grey, linestyle='--', label='Benchmark', linewidth=1.5)
# ax.set_xlabel('L2 Regularization')
# ax.set_ylabel('AIC')
# ax.set_xscale('log')
# ax.grid(True, alpha=0.3)
# ax.legend()
ax.set_xticklabels(['']*len(x_values))
plt.tight_layout()
plt.savefig(f'analysis/plots_model_evaluation/{study}_AIC.png', dpi=500)
plt.show()

# Plot 4: BIC values (excluding RNN and LSTM baselines due to high values)
fig4, ax = plt.subplots(figsize=(6, 5))
ax.plot(x_values, BIC_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=BIC_baseline, color=light_grey, linestyle=':', label='Baseline', linewidth=1.5)
ax.axhline(y=BIC_benchmark, color=dark_grey, linestyle='--', label='Benchmark', linewidth=1.5)
# ax.set_xlabel('L2 Regularization')
# ax.set_ylabel('BIC')
# ax.set_xscale('log')
# ax.grid(True, alpha=0.3)
# ax.legend()
ax.set_xticklabels(['']*len(x_values))
plt.tight_layout()
plt.savefig(f'analysis/plots_model_evaluation/{study}_BIC.png', dpi=500)
plt.show()



 
