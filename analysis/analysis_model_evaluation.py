import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import get_scores
from utils.setup_agents import setup_agent_rnn, setup_agent_spice, setup_agent_mcmc
from utils.convert_dataset import convert_dataset
from resources.bandits import get_update_dynamics, AgentQ
from benchmarking.hierarchical_bayes_numpyro import rl_model

import warnings
from sklearn.exceptions import ConvergenceWarning  # Import the specific warning type

warnings.filterwarnings("ignore", category=ConvergenceWarning)

drop_bad_participants = True

path_data = 'data/eckstein2022/eckstein2022.csv'
path_model_rnn = 'params/eckstein2022/params_eckstein2022.pkl'
path_model_benchmark = 'params/eckstein2022/params_eckstein2022_MODEL.nc'
path_model_baseline = 'params/eckstein2022/params_eckstein2022_ApBr.nc'

# path_data = 'data/parameter_recovery/data_16p_0.csv'
# path_model_rnn = 'params/parameter_recovery/params_16p_0.pkl'
# path_model_benchmark = None
# path_model_baseline = None

train_test_ratio = 0.8

models_benchmark = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']
# models_benchmark = ['ApAnBr']
dataset, experiment_list, _, _ = convert_dataset(path_data)

# ------------------------------------------------------------
# Setup of agents
# ------------------------------------------------------------

# setup baseline model
# old: win-stay-lose-shift -> very bad fit; does not bring the point that SPICE models are by far better than original ones
# new: Fitted ApBr model -> Tells the "true" story of how much better SPICE models can actually be by setting a good relative baseline
print("Setting up baseline agent...")
if path_model_baseline:
    agent_baseline = setup_agent_mcmc(path_model_baseline)
else:
    agent_baseline = [AgentQ(alpha_reward=0.3, beta_reward=1) for _ in range(len(dataset))]

n_parameters_baseline = 2

# setup benchmark models
if path_model_benchmark:
    print("Setting up benchmark agent...")
    agent_benchmark = {}
    for model in models_benchmark:
        agent_benchmark[model] = setup_agent_mcmc(path_model_benchmark.replace('MODEL', model))
    mapping_n_parameters_benchmark = {
        'ApBr': 2,
        'ApAnBr': 3,
        'ApBcBr': 3,
        'ApAcBcBr': 4,
        'ApAnBcBr': 4,
        'ApAnAcBcBr': 5,
    }
else:
     models_benchmark = []
n_parameters_benchmark = 0

# setup rnn agent
print("Setting up RNN agent...")
agent_rnn = setup_agent_rnn(
    path_model=path_model_rnn, 
    list_sindy_signals=['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'] + ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice'],
    )
n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

# setup spice agent
print("Setting up SPICE agent...")
rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
control_parameters = ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice']
sindy_library_setup = {
    'x_learning_rate_reward': ['c_reward', 'c_value_reward', 'c_value_choice'],
    'x_value_reward_not_chosen': ['c_value_choice'],
    'x_value_choice_chosen': ['c_value_reward'],
    'x_value_choice_not_chosen': ['c_value_reward'],
}
sindy_filter_setup = {
    'x_learning_rate_reward': ['c_action', 1, True],
    'x_value_reward_not_chosen': ['c_action', 0, True],
    'x_value_choice_chosen': ['c_action', 1, True],
    'x_value_choice_not_chosen': ['c_action', 0, True],
}
sindy_dataprocessing = None#{
#     'x_learning_rate_reward': [0, 0, 0],
#     'x_value_reward_not_chosen': [0, 0, 0],
#     'x_value_choice_chosen': [1, 1, 0],
#     'x_value_choice_not_chosen': [1, 1, 0],
#     'c_value_reward': [0, 0, 0],
#     'c_value_choice': [1, 1, 0],
# }

agent_spice = setup_agent_spice(
    path_model=path_model_rnn,
    path_data=path_data,
    rnn_modules=rnn_modules,
    control_parameters=control_parameters,
    sindy_library_setup=sindy_library_setup,
    sindy_filter_setup=sindy_filter_setup,
    sindy_dataprocessing=sindy_dataprocessing,
    sindy_library_polynomial_degree=1,
    regularization=0.1,
    threshold=0.05,
)
n_parameters_spice_all = agent_spice.count_parameters(mapping_modules_values={'x_learning_rate_reward': 'x_value_reward', 'x_value_reward_not_chosen': 'x_value_reward', 'x_value_choice_chosen': 'x_value_choice', 'x_value_choice_not_chosen': 'x_value_choice'})
n_parameters_spice = 0


# ------------------------------------------------------------
# Computation of metrics
# ------------------------------------------------------------

print('Running model evaluation...')
scores = np.zeros((4+len(models_benchmark), 3))
failed_attempts = 0
considered_trials = 0
index_participants_list, scores_spice_list, scores_baseline_list, scores_rnn_list, scores_benchmark_list, n_trials_list = [], [], [], [], [], []
for index_session in tqdm(range(len(dataset))):
    try:
        n_trials = len(experiment_list[index_session].choices)
        n_trials_test = int(n_trials * (1-train_test_ratio))
        # use whole session to include warm-up phase; make sure to exclude warm-up phase when computing metrics
        session = dataset.xs[index_session, :n_trials]
        choices = session[..., :agent_rnn._n_actions].cpu().numpy()
        choices_test = choices[-n_trials_test:]
        
        probs_test = get_update_dynamics(experiment=session, agent=agent_baseline[index_session])[1][-n_trials_test:]
        scores_baseline = np.array(get_scores(data=choices_test, probs=probs_test, n_parameters=n_parameters_baseline))
        scores_baseline_list.append(scores_baseline[0])
        
        probs_test = get_update_dynamics(experiment=session, agent=agent_spice)[1][-n_trials_test:]
        scores_spice = np.array(get_scores(data=choices_test, probs=probs_test, n_parameters=n_parameters_spice_all[index_session]))
        scores_spice_list.append(scores_spice[0])
        index_participants_list.append(index_session)
        
        probs_test = get_update_dynamics(experiment=session, agent=agent_rnn)[1][-n_trials_test:]
        scores_rnn = np.array(get_scores(data=choices_test, probs=probs_test, n_parameters=n_parameters_rnn))
        scores_rnn_list.append(scores_rnn[0])
        
        # get scores of all benchmark models but keep only the best one for each session
        if path_model_benchmark:
            scores_benchmark = []
            for index_model, model in enumerate(models_benchmark):
                probs_test = get_update_dynamics(experiment=session, agent=agent_benchmark[model][index_session])[1][-n_trials_test:]
                scores_benchmark_model = np.array(get_scores(data=choices_test, probs=probs_test, n_parameters=mapping_n_parameters_benchmark[model]))
                scores_benchmark.append(scores_benchmark_model)
                scores[4+index_model] += scores_benchmark_model
            scores_benchmark = np.stack(scores_benchmark)
            index_best_benchmark = np.argmin(scores_benchmark, axis=0)
            # take NLL as indicating score
            index_best_benchmark = index_best_benchmark[0]
            n_parameters_benchmark += mapping_n_parameters_benchmark[models_benchmark[index_best_benchmark]]
        
        n_trials_list.append(copy(n_trials_test))
        if np.exp(-scores_spice[0]/(n_trials_test*agent_rnn._n_actions)) < 0.6 and np.exp(-scores_rnn[0]/(n_trials_test*agent_rnn._n_actions)) > 0.63:
            raise ValueError(f'SPICE NLL ({scores_spice[0]}) is unplausibly high compared to RNN NLL ({scores_rnn[0]}). SPICE fitting seems bad parameterized. Skipping participant {index_session}.')
        
        n_parameters_spice += n_parameters_spice_all[index_session]
        
        # track scores
        scores[0] += scores_baseline
        if path_model_benchmark:
            scores[1] += scores_benchmark[index_best_benchmark]
        scores[2] += scores_rnn
        scores[3] += scores_spice
        
        # track number of trials
        considered_trials += n_trials_test
    except Exception as e:
        print(e)
        failed_attempts += 1

# ------------------------------------------------------------
# Post processing
# ------------------------------------------------------------

# print()
scores_all = np.concatenate((
    np.array(index_participants_list).reshape(-1, 1), 
    np.array(n_trials_list).reshape(-1, 1), 
    np.array(scores_baseline_list).reshape(-1, 1), 
    np.array(scores_rnn_list).reshape(-1, 1), 
    np.array(scores_spice_list).reshape(-1, 1),
    ), axis=-1)

import pandas as pd
pd.DataFrame(np.round(scores_all, 2), columns=['Participant', 'Trials', 'Baseline', 'RNN', 'SPICE']).to_csv('all_scores.csv')

# compute trial-level metrics (and NLL -> Likelihood)
avg_log_likelihood = -scores[:, :1] / (considered_trials * agent_rnn._n_actions)
avg_trial_likelihood = np.exp(avg_log_likelihood)

# compute average number of parameters
n_parameters_benchmark_single_models = [mapping_n_parameters_benchmark[model] for model in models_benchmark] if path_model_benchmark else []
n_parameters = np.array([
    n_parameters_baseline,
    n_parameters_benchmark/(len(dataset)-failed_attempts) if path_model_benchmark else 0, 
    n_parameters_rnn, 
    n_parameters_spice/(len(dataset)-failed_attempts),
    ]+n_parameters_benchmark_single_models)

scores = np.concatenate((avg_trial_likelihood, scores, n_parameters.reshape(-1, 1)), axis=1)

# ------------------------------------------------------------
# Printing model performance table
# ------------------------------------------------------------

print(f'Failed attempts: {failed_attempts}')

df = pd.DataFrame(
    data=scores,
    index=['Baseline', 'Benchmark', 'RNN', 'SPICE']+models_benchmark,
    columns = ('Trial Lik.', 'NLL', 'BIC', 'AIC', 'n_parameters'),
    )
print(df)