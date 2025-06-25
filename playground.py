import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from resources.bandits import AgentQ
from resources.rnn_utils import split_data_along_sessiondim
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.plotting import plot_session
from utils.convert_dataset import convert_dataset
from resources import rnn, sindy_utils

from benchmarking.benchmarking_lstm import setup_agent_lstm
from benchmarking.benchmarking_dezfouli2019 import setup_agent_mcmc as setup_agent_mcmc_dezfouli, gql_model
from benchmarking.benchmarking_eckstein2022 import setup_agent_mcmc as setup_agent_mcmc_eckstein, rl_model
from benchmarking.benchmarking_dezfouli2019_sgd import setup_agent_gql, Dezfouli2019GQL

# Your existing code
path_data = 'data/dezfouli2019/dezfouli2019.csv'
# path_mcmc = 'params/dezfouli2019/mcmc_dezfouli2019_gql.nc'
# path_rnn = 'params/eckstein2022/rnn_eckstein2022_FC_v1_ep1024.pkl'
# path_rnn_2 = 'params/eckstein2022/rnn_eckstein2022_FC_v2_ep1024.pkl'
# path_rnn_3 = 'params/eckstein2022/rnn_eckstein2022_no_l1_l2_0_0005_ep4096.pkl'
path_gql = 'params/dezfouli2019/gql_dezfouli2019.pkl'

path_spice = None#'params/eckstein2022/spice_eckstein2022_no_l1_l2_0_0005.pkl'

dataset = convert_dataset(path_data)[0]

agent_gql, n_parameters = setup_agent_gql(path_model=path_gql, model_config='PhiChiBetaKappaC', dimensions=2)

# agent_mcmc, n_parameters = setup_agent_mcmc_dezfouli(path_mcmc)

# agent_rnn = setup_agent_rnn(
#     class_rnn=rnn.RLRNN_eckstein2022_FC,
#     path_model=path_rnn,
# )

# agent_rnn_2 = setup_agent_rnn(
#     class_rnn=rnn.RLRNN_eckstein2022_FC,
#     path_model=path_rnn_2,
# )

# if path_spice is not None:
#     agent_spice = setup_agent_spice(
#         class_rnn=rnn.RLRNN_eckstein2022,
#         path_rnn=path_rnn,
#         path_spice=path_spice,
#     )

    # print('\n\nDiscovered SPICE models:\n')
    # for pid in [0, 1, 2, 3, 280, 285, 290, 291]:
    #     print(f'\nFor participant {pid}:\n')
    #     agent_spice.print_model(participant_id=pid)

fig, axs = plot_session(
    agents={
        # 'rnn': agent_rnn,
        # 'benchmark': agent_rnn_2,
        # 'sindy': agent_spice,
        'benchmark': agent_gql[0],
        },
    experiment=dataset.xs[291],
    display_choice=0
    )
plt.show()