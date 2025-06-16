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
from benchmarking.lstm_training import setup_agent_lstm, RLLSTM
from resources.rnn import RLRNN, RLRNN_eckstein2022, RLRNN_dezfouli2019, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_rearranged, RLRNN_dezfouli2019_blocks
from resources.sindy_utils import SindyConfig, SindyConfig_eckstein2022, SindyConfig_dezfouli2019, SindyConfig_eckstein2022_trials, SindyConfig_dezfouli2019_blocks
from resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim


# -------------------------------------------------------------------------------
# AGENT CONFIGURATIONS
# -------------------------------------------------------------------------------

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
# dataset = 'eckstein2022'
# models_benchmark = ['ApBr', 'ApBrAcfpBcf', 'ApBrAcfpBcfBch', 'ApAnBrBch', 'ApAnBrAcfpAcfnBcfBch', 'ApAnBrBcfBch']
# train_test_ratio = 0.8
# sindy_config = SindyConfig_eckstein2022
# rnn_class = RLRNN_eckstein2022
# additional_inputs = None
# -------------------- CONFIGURATION ECKSTEIN2022 w/ AGE --------------------
# rnn_class = RLRNN_meta_eckstein2022
# additional_inputs = ['age']

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
# dataset = 'dezfouli2019'
# train_test_ratio = [3, 6, 9]
# models_benchmark = ['ApBr', 'ApBrBch']
# sindy_config = SindyConfig_dezfouli2019
# rnn_class = RLRNN_dezfouli2019
# additional_inputs = []

# ------------------------ CONFIGURATION DEZFOULI2019 w/ blocks -----------------------
dataset = 'dezfouli2019'
train_test_ratio = [3, 6, 9]
models_benchmark = ['ApAnBrBcfAchBch']#['ApBr', 'ApBrBch', 'ApAnBrBcfAchBch']
sindy_config = SindyConfig_dezfouli2019
rnn_class = RLRNN_dezfouli2019
additional_inputs = []

# ------------------------- CONFIGURATION FILE PATHS ------------------------
use_test = True

path_data = f'data/{dataset}/{dataset}.csv'
# path_model_rnn = None#f'params/{dataset}/rnn_{dataset}_rldm_l1emb_0_001_l2_0_0005.pkl'
path_model_rnn = f'params/{dataset}/rnn_{dataset}_no_l1_l2_0.pkl'
path_model_spice = f'params/{dataset}/spice_{dataset}_no_l1_l2_0.pkl'
path_model_baseline = None#f'params/{dataset}/mcmc_{dataset}_ApBr.nc'
path_model_benchmark = None#f'params/{dataset}/mcmc_{dataset}_MODEL.nc' if len(models_benchmark) > 0 else None
path_model_benchmark_lstm = None#f'params/{dataset}/lstm_{dataset}_training_0_5.pkl'

# -------------------------------------------------------------------------------
# MODEL COMPARISON PIPELINE
# -------------------------------------------------------------------------------

dataset = convert_dataset(path_data, additional_inputs=additional_inputs)[0]
# use these participant_ids if not defined later
participant_ids = dataset.xs[:, 0, -1].unique().cpu().numpy()
    
# ------------------------------------------------------------
# Setup of agents
# ------------------------------------------------------------

print("Computing metrics on", 'test' if use_test else 'training', "data...")

# setup baseline model
# old: win-stay-lose-shift -> very bad fit; does not bring the point that SPICE models are by far better than original ones
# new: Fitted ApBr model -> Tells the "true" story of how much better SPICE models can actually be by setting a good relative baseline
print("Setting up baseline agent from file", path_model_baseline)
if path_model_baseline:
    agent_baseline = setup_agent_mcmc(path_model=path_model_baseline)
else:
    # agent_baseline = [AgentQ(alpha_reward=0.3, beta_reward=1) for _ in range(len(dataset))]
    agent_baseline = [AgentQ(alpha_reward=0., beta_reward=1, beta_choice=3) for _ in range(len(dataset))]

n_parameters_baseline = 2

# setup benchmark models
if path_model_benchmark:
    print("Setting up benchmark agent...")
    agent_benchmark = {}
    for model in models_benchmark:
        agent_benchmark[model] = setup_agent_mcmc(path_model=path_model_benchmark.replace('MODEL', model))
    mapping_n_parameters_benchmark = {model: sum([letter.isupper() for letter in model]) for model in models_benchmark}
    for model in mapping_n_parameters_benchmark:
        # reduce for one parameter if Bcf is in model name because that one is either 1 or 0 and not fitted
        if 'Bcf' in model:
            mapping_n_parameters_benchmark[model] -= 1
else:
    models_benchmark = []
n_parameters_benchmark = 0

if path_model_benchmark_lstm:
    print("Setting up benchmark LSTM agent...")
    agent_lstm = setup_agent_lstm(path_model=path_model_benchmark_lstm)
    n_parameters_lstm = sum(p.numel() for p in agent_lstm._model.parameters() if p.requires_grad)
else:
    n_parameters_lstm = 0
    
# setup rnn agent
if path_model_rnn is not None:
    print("Setting up RNN agent from file", path_model_rnn)
    agent_rnn = setup_agent_rnn(
        class_rnn=rnn_class,
        path_model=path_model_rnn, 
        list_sindy_signals=sindy_config['rnn_modules']+sindy_config['control_parameters'],
        )
    n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)
else:
    n_parameters_rnn = 0
    
# setup spice agent
if path_model_spice is not None:
    print("Setting up SPICE agent from file", path_model_spice)
    
    # get SPICE agent
    agent_spice = setup_agent_spice(
        class_rnn=rnn_class,
        path_rnn=path_model_rnn,
        path_spice=path_model_spice,
        path_data=path_data,
        rnn_modules=sindy_config['rnn_modules'],
        control_parameters=sindy_config['control_parameters'],
        sindy_library_setup=sindy_config['library_setup'],
        sindy_filter_setup=sindy_config['filter_setup'],
        sindy_dataprocessing=sindy_config['dataprocessing_setup'],
        sindy_library_polynomial_degree=1,
        regularization=0.1,
        threshold=0.05,
        filter_bad_participants=True,
    )

    # # get remaining participant_ids after removing badly fitted participants
    # participant_ids = agent_spice.get_participant_ids()
    # participant_ids_data = dataset.xs[:, 0, -1].unique().cpu().numpy()
    # if len(participant_ids) < len(participant_ids_data):
    #     removed_pids = []
    #     for pid_data in participant_ids_data:
    #         if not pid_data in participant_ids:
    #             removed_pids.append(pid_data)
    #     print(f"Removed participants due to bad SINDy fit: {removed_pids}")
n_parameters_spice = 0
    
# ------------------------------------------------------------
# Dataset splitting
# ------------------------------------------------------------

# split data into according to train_test_ratio
if isinstance(train_test_ratio, float):
    dataset_train, dataset_test = split_data_along_timedim(dataset, split_ratio=train_test_ratio)
    data_input = dataset.xs
    data_test = dataset.xs[..., :agent_baseline[0]._n_actions]
    # n_trials_test = dataset_test.xs.shape[1]
    
elif isinstance(train_test_ratio, list) or isinstance(train_test_ratio, tuple):
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, list_test_sessions=train_test_ratio)
    if not use_test:
        dataset_test = dataset_train
    data_input = dataset_test.xs
    data_test = dataset_test.xs[..., :agent_baseline[0]._n_actions]
    
else:
    raise TypeError("train_test_raio must be either a float number or a list of integers containing the session/block ids which should be used as test sessions/blocks")
n_trials_test = dataset_test.xs.shape[1]

# ------------------------------------------------------------
# Computation of metrics
# ------------------------------------------------------------

print('Running model evaluation...')
scores = np.zeros((5+len(models_benchmark), 3))

failed_attempts = 0
considered_trials = 0

metric_participant = np.zeros((len(scores), len(dataset_test)))
best_benchmarks_participant, considered_trials_participant = np.array(['' for _ in range(len(dataset_test))]), np.zeros(len(dataset_test))

for index_data in tqdm(range(len(dataset_test))):
    try:
        # use whole session to include warm-up phase; make sure to exclude warm-up phase when computing metrics
        pid = dataset_test.xs[index_data, 0, -1].item()
        
        if not pid in participant_ids:
            print(f"Skipping participant {index_data} because they could not be found in the SPICE participants. Probably due to prior filtering of badly fitted participants.")
            continue
        
        # Baseline model
        probs_baseline = get_update_dynamics(experiment=data_input[index_data], agent=agent_baseline[index_data])[1]
        
        # get number of actual trials
        n_trials = len(probs_baseline)
        data_ys = data_test[index_data, :n_trials].cpu().numpy()
        
        if isinstance(train_test_ratio, float):
            n_trials_test = int(n_trials*(1-train_test_ratio))
            if use_test:
                index_start = n_trials - n_trials_test
                index_end = n_trials
            else:
                index_start = 0
                index_end = n_trials - n_trials_test
        else:
            index_start = 0
            index_end = n_trials
        
        scores_baseline = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_baseline[index_start:index_end], n_parameters=n_parameters_baseline))
        metric_participant[0, index_data] += scores_baseline[0]
        
        # get scores of all mcmc benchmark models but keep only the best one for each session
        if path_model_benchmark:
            scores_benchmark = np.zeros((len(models_benchmark), 3))
            for index_model, model in enumerate(models_benchmark):
                probs_benchmark = get_update_dynamics(experiment=data_input[index_data], agent=agent_benchmark[model][index_data])[1]
                scores_benchmark[index_model] += np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_benchmark[index_start:index_end], n_parameters=mapping_n_parameters_benchmark[model]))
            index_best_benchmark = np.argmin(scores_benchmark, axis=0)[1] # index 0 -> NLL is indicating metric
            n_parameters_benchmark += mapping_n_parameters_benchmark[models_benchmark[index_best_benchmark]]
            best_benchmarks_participant[index_data] += models_benchmark[index_best_benchmark]
            metric_participant[1, index_data] += scores_benchmark[index_best_benchmark, 0]
            metric_participant[5:, index_data] += scores_benchmark[:, 0]
        
        # Benchmark LSTM
        if path_model_benchmark_lstm:
            probs_lstm = get_update_dynamics(experiment=data_input[index_data], agent=agent_lstm)[1]
            scores_lstm = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_lstm[index_start:index_end], n_parameters=n_parameters_lstm))
            metric_participant[2, index_data] += scores_lstm[0]
            
        # SPICE-RNN
        if path_model_rnn is not None:
            probs_rnn = get_update_dynamics(experiment=data_input[index_data], agent=agent_rnn)[1]
            scores_rnn = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_rnn[index_start:index_end], n_parameters=n_parameters_rnn))
            metric_participant[3, index_data] = scores_rnn[0]
            
        # SPICE
        if path_model_spice is not None:
            additional_inputs_embedding = data_input[0, agent_spice._n_actions*2:-3]
            agent_spice.new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
            n_params_spice = agent_spice.count_parameters()[pid]
            
            probs_spice = get_update_dynamics(experiment=data_input[index_data], agent=agent_spice)[1]
            scores_spice = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_spice[index_start:index_end], n_parameters=n_params_spice))
            n_parameters_spice += n_params_spice
            metric_participant[4, index_data] = scores_spice[0]
        
        considered_trials_participant[index_data] += index_end - index_start
        considered_trials += index_end - index_start
        
        # track scores
        scores[0] += scores_baseline
        if path_model_benchmark:
            scores[1] += scores_benchmark[index_best_benchmark]
            scores[5:] += scores_benchmark
        if path_model_benchmark_lstm:
            scores[2] += scores_lstm
        if path_model_rnn is not None:
            scores[3] += scores_rnn
        if path_model_spice is not None:
            scores[4] += scores_spice
        
    except Exception as e:  
        # print(e)
        raise e
        failed_attempts += 1

# ------------------------------------------------------------
# Post processing
# ------------------------------------------------------------

if path_model_benchmark:
    # print how often each benchmark model was the best one
    from collections import Counter
    occurrences = Counter(best_benchmarks_participant)
    print("Counter for each benchmark model being the best one:")
    print(occurrences)

# compute trial-level metrics (and NLL -> Likelihood)
scores = scores / (considered_trials)# * agent_baseline[0]._n_actions)
avg_trial_likelihood = np.exp(- scores[:, 0])
# avg_trial_likelihood = np.exp(- scores_participant.T / (considered_trials_participant.reshape(-1, 1) * agent_baseline[0]._n_actions))
# scores[:, 0] = scores[:, 0] / len(dataset)

metric_participant_std = (metric_participant/considered_trials_participant).std(axis=1)
avg_trial_likelihood_participant = np.exp(- metric_participant / considered_trials_participant)
avg_trial_likelihood_participant_std = avg_trial_likelihood_participant.std(axis=1)

# pd.DataFrame(data=np.concatenate((np.array(best_benchmarks_participant).reshape(-1, 1), avg_trial_likelihood), axis=1), columns=['benchmark model', 'baseline','benchmark', 'lstm', 'rnn', 'spice']+models_benchmark).to_csv('best_scores_benchmark.csv')

# compute average number of parameters
n_parameters_benchmark_single_models = [mapping_n_parameters_benchmark[model] for model in models_benchmark] if path_model_benchmark else []
n_parameters = np.array([
    n_parameters_baseline,
    n_parameters_benchmark/(len(dataset)-failed_attempts) if path_model_benchmark else 0, 
    n_parameters_lstm,
    n_parameters_rnn, 
    n_parameters_spice/(len(dataset)-failed_attempts),
    ]+n_parameters_benchmark_single_models)

scores = np.concatenate((avg_trial_likelihood.reshape(-1, 1), avg_trial_likelihood_participant_std.reshape(-1, 1), scores[:, :1], metric_participant_std.reshape(-1, 1), scores[:, 1:], n_parameters.reshape(-1, 1)), axis=1)


# ------------------------------------------------------------
# Printing model performance table
# ------------------------------------------------------------

print(f'Failed attempts: {failed_attempts}')

df = pd.DataFrame(
    data=scores,
    index=['Baseline', 'Benchmark', 'LSTM', 'RNN', 'SPICE']+models_benchmark,
    columns = ('Trial Lik.', '(std)', 'NLL', '(std)', 'BIC', 'AIC', 'n_parameters'),
    )
print(df)