import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_mcmc
from benchmarking.lstm_training import setup_agent_lstm
from utils.plotting import plot_session
from resources.rnn import RLRNN_dezfouli2019
from resources.sindy_utils import SindyConfig_dezfouli2019
from resources.bandits import AgentQ


agent_q = AgentQ(beta_choice=1, beta_reward=3)
agent_mcmc = setup_agent_mcmc('params/hbi_test_ApBrBch.nc')

dataset = convert_dataset('data/data_128p_0.csv')[0]

fig, axs = plot_session({'groundtruth': agent_q, 'benchmark': agent_mcmc[0]}, experiment=dataset.xs[0], display_choice=0)
plt.show()