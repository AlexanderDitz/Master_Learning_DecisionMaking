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


# \toprule
# &$n_\text{parameters}$&$(\sigma)$&$\bar{\mathcal{L}}$&($\sigma$)&NLL&AIC&BIC\\
# \midrule
# Baseline&2&0&0.67622&0.12087&0.39124&0.81364&0.86432\\
# Benchmark&4&0&0.70341&0.13075&0.35182&0.74656&0.84791\\
# LSTM&1442&0&0.69989&0.12470&0.35684&28.30485&64.84373\\
# \midrule
# $l_2=0$&&&&&&&\\
# RNN&547&0&0.69346&0.14281&0.36606&11.10921&24.96966\\
# SPICE&13.31&3.67&0.61740&0.16760&0.48224&1.09541&1.43259\\
# \midrule
# $l_2=0.00001$&&&&&&&\\
# RNN&547&0&0.69620&0.14050&0.36211&11.07344&24.93389\\
# SPICE&14.43&2.96&0.68266&0.14289&0.38176&&0.95644&1.32211\\
# \midrule
# $l_2=0.00005$&&&&&&&\\
# RNN&547&0&0.70280&0.13696&0.35269&11.09775&24.95819\\
# SPICE&15.17&2.19&0.70148&0.13659&0.35457&0.93909&1.32360\\
# \midrule
# $l_2=0.0001$&&&&&&&\\
# RNN&547&0&0.70560&0.13410&0.34870&11.09763&24.95808\\
# SPICE&14.82&1.36&0.70348&0.13228&0.35172&0.93633&1.31191\\
# \midrule
# $l_2=0.0005$&&&&&&&\\
# RNN&547&0&0.70459&0.12051&0.35014&11.15791&25.01836\\
# SPICE&12.83&1.96&0.70288&0.12018&0.35257&0.94384&1.26878\\
# \midrule
# $l_2=0.001$&&&&&&&\\
# RNN&547&0&0.69702&0.11429&0.36093&11.18507&25.04551\\
# SPICE&11.41&0.59&0.69577&0.11442&0.36273&0.94101&1.23005\\
# \bottomrule


L_baseline, NLL_baseline, AIC_baseline, BIC_baseline = 0.67622, 0.39124, 0.81364, 0.86432
L_benchmark, NLL_benchmark, AIC_benchmark, BIC_benchmark = 0.70341, 0.35182, 0.74656, 0.84791
L_LSTM, NLL_LSTM, AIC_LSTM, BIC_LSTM = 0.69989, 0.35684, 28.30485, 64.84373

# L̄ (L_bar) values
L_RNN = [0.69346, 0.69620, 0.70280, 0.70560, 0.70459, 0.69702]
L_SPICE = [0.61740, 0.68266, 0.70148, 0.70348, 0.70288, 0.69577]

# NLL (Negative Log-Likelihood) values
NLL_RNN = [0.36606, 0.36211, 0.35269, 0.34870, 0.35014, 0.36093]
NLL_SPICE = [0.48224, 0.38176, 0.35457, 0.35172, 0.35257, 0.36273]

# AIC (Akaike Information Criterion) values
AIC_RNN = [11.10921, 11.07344, 11.09775, 11.09763, 11.15791, 11.18507]
AIC_SPICE = [1.09541, 0.95644, 0.93909, 0.93633, 0.94384, 0.94101]

# BIC (Bayesian Information Criterion) values
BIC_RNN = [24.96966, 24.93389, 24.95819, 24.95808, 25.01836, 25.04551]
BIC_SPICE = [1.43259, 1.32211, 1.32360, 1.31191, 1.26878, 1.23005]

# L2 regularization values (for reference)
l2_values = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]



# Colors
orange = '#FF8C00'
pink = '#FF69B4'
dark_grey = '#2F2F2F'

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Plot L̄ values
ax = axes[0, 0]
ax.plot(l2_values, L_RNN, 'x-', color=orange, linestyle='--', label='RNN', markersize=8, linewidth=2)
ax.plot(l2_values, L_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=L_baseline, color=dark_grey, linestyle='--', alpha=0.8)
ax.axhline(y=L_benchmark, color=dark_grey, linestyle='--', alpha=0.8)
ax.axhline(y=L_LSTM, color=dark_grey, linestyle='--', alpha=0.8)
ax.text(max(l2_values)*1.02, L_baseline, 'Baseline', va='center', color=dark_grey, fontsize=9)
ax.text(max(l2_values)*1.02, L_benchmark, 'Benchmark', va='center', color=dark_grey, fontsize=9)
ax.text(max(l2_values)*1.02, L_LSTM, 'LSTM', va='center', color=dark_grey, fontsize=9)
ax.set_xlabel('L2 Regularization')
ax.set_ylabel(r'$\bar{\mathcal{L}}$')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot NLL values
ax = axes[0, 1]
ax.plot(l2_values, NLL_RNN, 'x-', color=orange, linestyle='--', label='RNN', markersize=8, linewidth=2)
ax.plot(l2_values, NLL_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=NLL_baseline, color=dark_grey, linestyle='--', alpha=0.8)
ax.axhline(y=NLL_benchmark, color=dark_grey, linestyle='--', alpha=0.8)
ax.axhline(y=NLL_LSTM, color=dark_grey, linestyle='--', alpha=0.8)
ax.text(max(l2_values)*1.02, NLL_baseline, 'Baseline', va='center', color=dark_grey, fontsize=9)
ax.text(max(l2_values)*1.02, NLL_benchmark, 'Benchmark', va='center', color=dark_grey, fontsize=9)
ax.text(max(l2_values)*1.02, NLL_LSTM, 'LSTM', va='center', color=dark_grey, fontsize=9)
ax.set_xlabel('L2 Regularization')
ax.set_ylabel('NLL')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot AIC values (excluding RNN and LSTM baselines due to high values)
ax = axes[1, 0]
ax.plot(l2_values, AIC_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=AIC_baseline, color=dark_grey, linestyle='--', alpha=0.8)
ax.axhline(y=AIC_benchmark, color=dark_grey, linestyle='--', alpha=0.8)
ax.text(max(l2_values)*1.02, AIC_baseline, 'Baseline', va='center', color=dark_grey, fontsize=9)
ax.text(max(l2_values)*1.02, AIC_benchmark, 'Benchmark', va='center', color=dark_grey, fontsize=9)
ax.set_xlabel('L2 Regularization')
ax.set_ylabel('AIC')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot BIC values (excluding RNN and LSTM baselines due to high values)
ax = axes[1, 1]
ax.plot(l2_values, BIC_SPICE, 'x-', color=pink, linestyle='--', label='SPICE', markersize=8, linewidth=2)
ax.axhline(y=BIC_baseline, color=dark_grey, linestyle='--', alpha=0.8)
ax.axhline(y=BIC_benchmark, color=dark_grey, linestyle='--', alpha=0.8)
ax.text(max(l2_values)*1.02, BIC_baseline, 'Baseline', va='center', color=dark_grey, fontsize=9)
ax.text(max(l2_values)*1.02, BIC_benchmark, 'Benchmark', va='center', color=dark_grey, fontsize=9)
ax.set_xlabel('L2 Regularization')
ax.set_ylabel('BIC')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()



 
