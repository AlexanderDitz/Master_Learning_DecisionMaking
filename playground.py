import matplotlib.pyplot as plt
from utils.setup_agents import setup_agent_spice, setup_agent_rnn
from utils.plotting import plot_session
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import split_data_along_timedim


participant_id = 195
path_data = 'data/eckstein2022/eckstein2022.csv'
path_rnn = 'params/eckstein2022/rnn_eckstein2022_reward.pkl'
path_spice = 'params/eckstein2022/spice_eckstein2022_reward.pkl'

rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
control_parameters = ['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice']
sindy_library_setup = {
    'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
    'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
    'x_value_choice_chosen': ['c_value_reward'],
    'x_value_choice_not_chosen': ['c_value_reward'],
}
sindy_filter_setup = {
    'x_learning_rate_reward': ['c_action', 1, True],
    'x_value_reward_not_chosen': ['c_action', 0, True],
    'x_value_choice_chosen': ['c_action', 1, True],
    'x_value_choice_not_chosen': ['c_action', 0, True],
}

agent_rnn = setup_agent_rnn(
    path_model=path_rnn,
    list_sindy_signals=rnn_modules+control_parameters,
)

agent_spice = setup_agent_spice(
    path_rnn=path_rnn,
    path_data=path_data,
    path_spice=path_spice,
    rnn_modules=rnn_modules,
    control_parameters=control_parameters,
    sindy_library_setup=sindy_filter_setup,
    sindy_filter_setup=sindy_filter_setup,
    sindy_dataprocessing=None,
    sindy_library_polynomial_degree=1,
)

dataset = split_data_along_timedim(convert_dataset(file=path_data)[0], split_ratio=0.8)[0]

plot_session(
    agents = {
        'rnn': agent_rnn,
        'sindy': agent_spice, 
    },
    experiment = dataset.xs[participant_id],
)
plt.show()