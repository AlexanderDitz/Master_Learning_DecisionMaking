import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import get_scores
from utils.setup_agents import setup_agent_rnn
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentQ, get_update_dynamics

# data = 'data/parameter_recovery_participants/data_128p_0.csv'
# model = 'params/parameter_recovery_participants/params_128p_0.pkl'

data = 'data/sugawara2021/sugawara2021.csv'
model = 'params/sugawara2021/params_sugawara2021.pkl'

dataset, experiment_list, _, _ = convert_dataset(data)

# setup rnn agent for comparison
agent_rnn = setup_agent_rnn(
    path_model=model, 
    list_sindy_signals=['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'] + ['c_action', 'c_reward', 'c_value_reward'],
    )
n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

# baseline model (win-stay-lose-shift)
agent_baseline = AgentQ(alpha_reward=1, beta_reward=10.)

scores = np.zeros((2, 3))
failed_attempts = 0
for index_session in tqdm(range(len(dataset))):
    # try:
    n_trials = len(experiment_list[index_session].choices)
    session = dataset.xs[index_session, :n_trials]
    choices = session[..., :agent_rnn._n_actions].cpu().numpy()
    
    probs = get_update_dynamics(experiment=session, agent=agent_baseline)[1]
    scores_rl = np.array(get_scores(data=choices, probs=probs, n_parameters=0))
    
    probs = get_update_dynamics(experiment=session, agent=agent_rnn)[1]
    scores_rnn = np.array(get_scores(data=choices, probs=probs, n_parameters=n_parameters_rnn))
    
    scores[0] += scores_rl
    scores[1] += scores_rnn
    # except:
    #     failed_attempts += 1

df = pd.DataFrame(
    data=scores,
    index=['Baseline', 'RNN'],
    columns = ('NLL', 'BIC', 'AIC'),
    )

# print(f'Number of ignored sessions due to SINDy error: {n_sessions - len(index_sindy_valid)}')
print(f'Failed attempts: {failed_attempts}')
print(df)
