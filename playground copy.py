import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from jax import random
import jax.numpy as jnp
import arviz as az
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi, summary, gelman_rubin, effective_sample_size, autocorrelation
from typing import Dict, Any, Optional, List
import pickle

from resources.bandits import AgentQ, get_update_dynamics, create_dataset, BanditsDrift, AgentNetwork
from resources.rnn_utils import split_data_along_sessiondim
from resources.rnn_training import fit_model
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.plotting import plot_session
from utils.convert_dataset import convert_dataset
from resources import rnn, sindy_utils, rnn_utils
from resources.model_evaluation import log_likelihood
from resources.rnn import RLRNN_eckstein2022

from benchmarking.benchmarking_lstm import setup_agent_lstm
from benchmarking import benchmarking_eckstein2022, benchmarking_dezfouli2019


# dataset = convert_dataset('data/q_agent_.csv')[0]

agent = AgentQ(alpha_reward=0.3, beta_reward=3.)
env = BanditsDrift(0.2)
dataset = create_dataset(agent, env, 100, 100)[0]

rnn = RLRNN_eckstein2022(2, 128)
optimizer = torch.optim.Adam(rnn.parameters(), 0.01)
rnn, optimizer, _ = fit_model(rnn, dataset, None, optimizer, 0, 1024)

agent_rnn = AgentNetwork(rnn)

fig, axs = plot_session({'groundtruth': agent, 'rnn': agent_rnn}, dataset.xs[0])
plt.show()