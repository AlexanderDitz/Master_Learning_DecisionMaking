import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_mcmc, setup_agent_spice
from benchmarking.lstm_training import setup_agent_lstm
from utils.plotting import plot_session
from resources.rnn import RLRNN_dezfouli2019, RLRNN_eckstein2022
from resources.sindy_utils import SindyConfig_dezfouli2019, SindyConfig_eckstein2022
from resources.bandits import AgentQ
from resources.rnn_utils import split_data_along_sessiondim
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session


# Your existing code
path_data = 'data/eckstein2022/eckstein2022.csv'
path_rnn = 'params/eckstein2022/rnn_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'
path_spice = 'params/eckstein2022/spice_eckstein2022_rldm_l1emb_0_001_l2_0_0005.pkl'

participant_id = 2

sindy_config = SindyConfig_eckstein2022


dataset = convert_dataset(path_data)[0]

agent_rnn = setup_agent_rnn(
    class_rnn=RLRNN_eckstein2022,
    path_model=path_rnn,
    list_sindy_signals=sindy_config['rnn_modules']+sindy_config['control_parameters'],
)

agent_spice = setup_agent_spice(
        class_rnn=RLRNN_eckstein2022,
        path_rnn=path_rnn,
        path_spice=path_spice,
        path_data=path_data,
        rnn_modules=sindy_config['rnn_modules'],
        control_parameters=sindy_config['control_parameters'],
        sindy_library_setup=sindy_config['library_setup'],
        sindy_filter_setup=sindy_config['filter_setup'],
        sindy_dataprocessing=sindy_config['dataprocessing_setup'],
        sindy_library_polynomial_degree=1,
    )

agent_spice.new_sess(participant_id=participant_id)
modules = agent_spice.get_modules()
for module in modules:
    agent_spice._model.submodules_sindy[module][participant_id].print()

fig, axs = plot_session({'sindy': agent_spice, 'rnn': agent_rnn}, dataset.xs[participant_id])
plt.savefig('plt_rldm.png', dpi=500)
# plt.show()