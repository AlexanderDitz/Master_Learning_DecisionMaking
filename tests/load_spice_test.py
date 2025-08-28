import sys
import os

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session
from utils.setup_agents import setup_agent_spice, setup_agent_rnn
from resources.rnn import RLRNN, RLRNN_eckstein2022, RLRNN_meta_eckstein2022, RLRNN_dezfouli2019
from resources.sindy_utils import load_spice, SindyConfig, SindyConfig_eckstein2022, SindyConfig_dezfouli2019

model_rnn = 'params/eckstein2022/rnn_eckstein2022_age.pkl'
model_spice = 'params/eckstein2022/spice_eckstein2022_age.pkl'
data = 'data/eckstein2022/eckstein2022_age.csv'
class_rnn = RLRNN_meta_eckstein2022
sindy_config = SindyConfig_eckstein2022
additional_inputs = ['age']

participant_id = 1

# here starts the testing of the loading functionality
agent_spice = setup_agent_spice(
    class_rnn=class_rnn, 
    path_spice=model_spice,
    path_rnn=model_rnn, 
    path_data=data,
    rnn_modules=sindy_config['rnn_modules'],
    control_parameters=sindy_config['control_parameters'],
    sindy_library_setup=sindy_config['library_setup'],
    sindy_filter_setup=sindy_config['filter_setup'],
    sindy_dataprocessing=sindy_config['dataprocessing_setup'],
    sindy_library_polynomial_degree=1,
)

agent_rnn = setup_agent_rnn(
    class_rnn=class_rnn,
    path_rnn=model_rnn,
    list_sindy_signals=sindy_config['rnn_modules']+sindy_config['control_parameters'],
)

dataset = convert_dataset(file=data, additional_inputs=additional_inputs)[0]

agent_spice.new_sess(participant_id=participant_id, additional_embedding_inputs=np.array(0.5))

modules = agent_spice.get_modules()
print("\nFitted modules for participant", participant_id)
for module in modules:
    modules[module][participant_id].print()
fig, axs = plot_session(experiment=dataset.xs[participant_id], agents={'rnn': agent_rnn, 'sindy': agent_spice})
plt.show()