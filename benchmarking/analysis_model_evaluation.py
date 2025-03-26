import os
import sys

import numpy as np
import pickle
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import get_scores
from resources.rnn_utils import DatasetRNN
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_benchmark_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentQ, get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model

burnin = 0
session_id = None
participant_emb = True,

# data = 'data/2arm/sugawara2021_143_processed.csv'
# model_rnn = 'params/benchmarking/rnn_sugawara.pkl'
# model_benchmark = 'benchmarking/params/sugawara2021_143/hierarchical/traces.nc'
# # model_benchmark = 'benchmarking/params/sugawara2021_143/traces_test.nc'
# results = 'benchmarking/results/results_sugawara.csv'

# data = 'data/2arm/eckstein2022_291_processed.csv'
# model_rnn = 'params/benchmarking/rnn_eckstein.pkl'
# model_benchmark = 'benchmarking/params/eckstein2022_291/traces.nc'
# results = 'benchmarking/results/results_eckstein.csv'

data='data/parameter_recovery_participants/data_128p_0.csv'
model_rnn='params/parameter_recovery_participants/params_128p_0_NoID.pkl'
model_benchmark = 'benchmarking/params/traces_test.nc'
results = 'benchmarking/results/results_study_recovery_stepperseverance_rldm_256p_0.csv'

# models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']
# models = ['ApAcBcBr']
models = []

# sindy configuration
rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
control_parameters = ['c_action', 'c_reward', 'c_value_reward']
sindy_library_polynomial_degree = 2

sindy_library_setup = {
    'x_learning_rate_reward': ['c_reward', 'c_value_reward'],
}

sindy_filter_setup = {
    'x_learning_rate_reward': ['c_action', 1, True],
    'x_value_reward_not_chosen': ['c_action', 0, True],
    'x_value_choice_chosen': ['c_action', 1, True],
    'x_value_choice_not_chosen': ['c_action', 0, True],
}

sindy_dataprocessing = None

# load data
dataset = convert_dataset(data)[0]
n_sessions = len(dataset.xs)
if isinstance(session_id, int):
    dataset_xs = dataset.xs[session_id][None]
    dataset_ys = dataset.ys[session_id][None]
    dataset = DatasetRNN(dataset_xs, dataset_ys)
    
# setup rnn agent for comparison
agent_rnn = setup_agent_rnn(
    path_model=model_rnn, 
    list_sindy_signals=rnn_modules+control_parameters
    )
n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

# setup sindy agent and get number of sindy coefficients which are not 0
agent_sindy = setup_agent_sindy(
    path_model=model_rnn, 
    path_data=data,
    rnn_modules=rnn_modules,
    control_parameters=control_parameters,
    sindy_library_polynomial_degree=sindy_library_polynomial_degree,
    sindy_library_setup=sindy_library_setup,
    sindy_filter_setup=sindy_filter_setup,
    sindy_dataprocessing=sindy_dataprocessing,
    participant_id=session_id,
    )
n_parameters_sindy = agent_sindy.count_parameters()

# setup AgentQ model with values from sugawara paper as baseline
agent_rl = AgentQ(alpha_reward=0.25, beta_reward=1.)#(alpha_reward=.45, beta_reward=.19, alpha_choice=0.41, beta_choice=1.10)

agent_mcmc = {}
for model in models:
    with open(model_benchmark.split('.')[0] + '_' + model + '.nc', 'rb') as file:
        mcmc = pickle.load(file)
    # mcmc.print_summary()
    parameters = {
        'alpha_pos': 1,
        'alpha_neg': -1,
        'alpha_c': 1,
        'beta_c': 0,
        'beta_r': 1,
    }
    n_parameters_mcmc = 0
    # mcmc.print_summary()
    params_mcmc = []
    for p in mcmc.get_samples():
        if not '_mean' in p and not '_std' in p:
            parameters[p] = np.mean(np.array(mcmc.get_samples()[p][burnin:]), axis=0)
            if not isinstance(parameters[p], np.ndarray):
                parameters[p] = np.full(n_sessions, parameters[p])
            n_parameters_mcmc += 1
            params_mcmc.append(p)
    
    # make all parameters an array that where not in hierarchical mcmc model to match shape of mcmc parameters
    for p in parameters:
        if not p in params_mcmc:
            parameters[p] = np.full(n_sessions, parameters[p])
    
    # in case of symmetric learning rates:
    if np.mean(parameters['alpha_neg']) == -1:
        parameters['alpha_neg'] = parameters['alpha_pos']

    agents = []
    for i in range(n_sessions):
        params_i = {p: parameters[p][i] for p in parameters}
        agents.append(setup_benchmark_q_agent(params_i))
    
    agent_mcmc[model] = (agents, n_parameters_mcmc)

# get scores by all agents
scores = np.zeros((3+len(models), 3))
failed_attempts = 0
for index_session, session in enumerate(dataset):
    try:
        choices = session[0][..., :agent_rl._n_actions].cpu().numpy()
        participant_id = session[0][-1, -1].int().item()
        
        probs = get_update_dynamics(experiment=session[0], agent=agent_rl)[1]
        scores_rl = np.array(get_scores(data=choices, probs=probs, n_parameters=2))
        
        probs = get_update_dynamics(experiment=session[0], agent=agent_rnn)[1]
        scores_rnn = np.array(get_scores(data=choices, probs=probs, n_parameters=n_parameters_rnn))
        
        probs = get_update_dynamics(experiment=session[0], agent=agent_sindy)[1]
        scores_sindy = np.array(get_scores(data=choices, probs=probs, n_parameters=n_parameters_sindy[participant_id]))

        scores_model = {}
        for index_model, model in enumerate(agent_mcmc):
            probs = get_update_dynamics(experiment=session[0], agent=agent_mcmc[model][0][index_session])[1]
            scores_model[model] = np.array(get_scores(data=choices, probs=probs, n_parameters=agent_mcmc[model][1][index_session]))
            scores[1+index_model] += scores_model[model]
        scores[0] += scores_rl
        scores[-2] += scores_rnn
        scores[-1] += scores_sindy
    except:
        failed_attempts += 1


# print('Get scores by SINDy...')
# df = get_scores_array(dataset, [agent_sindy[id] for id in agent_sindy], n_parameters_sindy)
# df.to_csv(results.replace('.', '_sindy.'), index=False)
# # get sessions where sindy recovered a weird equation leading to exploding values
# index_sindy_valid = (1-df['NLL'].isna()).astype(bool)
# data[-1] = np.array((df['NLL'].values[index_sindy_valid].sum(), df['AIC'].values[index_sindy_valid].sum(), df['BIC'].values[index_sindy_valid].sum()))

# print('Get LL by RL-Baseline...')
# df = get_scores_array(dataset, [agent_rl]*len(dataset), [2]*len(dataset))
# # data[0] = np.array((df['NLL'].values[index_sindy_valid].sum(), df['AIC'].values[index_sindy_valid].sum(), df['BIC'].values[index_sindy_valid].sum()))
# data[0] = np.array((df['NLL'].values.sum(), df['AIC'].values.sum(), df['BIC'].values.sum()))

# print('Get LL by RNN...')
# df = get_scores_array(dataset, [agent_rnn]*len(dataset), [n_parameters_rnn]*len(dataset))
# df.to_csv(results.replace('.', '_rnn.'), index=False)
# data[-2] = np.array((df['NLL'].values[index_sindy_valid].sum(), df['AIC'].values[index_sindy_valid].sum(), df['BIC'].values[index_sindy_valid].sum()))

# for i in range(1, len(models)+1):
#     key = list(agent_mcmc.keys())[i-1]
#     print(f'Get LL by Benchmark ({key})...')
#     df = get_scores_array(dataset, agent_mcmc[key][0], agent_mcmc[key][1])
#     df.to_csv(results.replace('.', '_'+key+'.'), index=False)
#     data[i] = np.array((df['NLL'].values[index_sindy_valid].sum(), df['AIC'].values[index_sindy_valid].sum(), df['BIC'].values[index_sindy_valid].sum()))

df = pd.DataFrame(
    data=scores,
    index=['RL']+models+['RNN', 'SINDy'],
    columns = ('NLL', 'BIC', 'AIC'),
    )

# print(f'Number of ignored sessions due to SINDy error: {n_sessions - len(index_sindy_valid)}')
print(f'Failed attempts: {failed_attempts}')
print(df)