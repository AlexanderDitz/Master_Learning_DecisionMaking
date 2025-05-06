import sys
import os

import matplotlib.pyplot as plt
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session

model = 'params/eckstein2022/rnn_eckstein2022.pkl'
data = 'data/eckstein2022/eckstein2022.csv'

agent_spice, features, loss = pipeline_sindy.main(
    
    # data='data/parameter_recovery/data_32p_0.csv',
    # model='params/parameter_recovery/params_32p_0.pkl',
    
    model = model,
    data = data,
    
    # general recovery parameters
    participant_id=0,
    filter_bad_participants=True,
    
    # sindy parameters
    polynomial_degree=1,
    optimizer_alpha=0.1,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=1,
    verbose=True,
    
    # generated training dataset parameters
    n_actions=2,
    sigma=0.2,
    beta_reward=1.,
    alpha=0.25,
    alpha_penalty=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=True,
    get_loss=True,
)
print(loss)

# here starts the testing of the loading functionality
agent_spice_preload = deepcopy(agent_spice)
agent_spice.load('params/eckstein2022/spice_eckstein2022.pkl')

dataset = convert_dataset(file=data)[0]
fig, axs = plot_session(experiment=dataset.xs[0], agents={'groundtruth': agent_spice_preload, 'sindy': agent_spice})
plt.show()