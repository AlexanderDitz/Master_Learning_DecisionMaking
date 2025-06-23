import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from resources.rnn import RLRNN_dezfouli2019, RLRNN_eckstein2022
from resources.sindy_utils import SindyConfig_dezfouli2019, SindyConfig_eckstein2022
from resources.bandits import AgentQ
from resources.rnn_utils import split_data_along_sessiondim
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.plotting import plot_session
from utils.convert_dataset import convert_dataset
from benchmarking.benchmarking_lstm import setup_agent_lstm
from benchmarking.benchmarking_dezfouli2019 import setup_agent_mcmc, gql_model

# Your existing code
path_data = 'data/eckstein2022/eckstein2022.csv'
path_model = 'params/dezfouli2019/mcmc_dezfouli2019_baseline.nc'
# path_rnn = 'params/dezfouli2019/rnn_dezfouli2019_no_l1_l2_0.pkl'

agent_mcmc = setup_agent_mcmc(path_model)
dataset = convert_dataset(path_data, )[0]

fig, axs = plot_session(agents={'benchmark': agent_mcmc[0][0]}, experiment=dataset.xs[0])
plt.show()