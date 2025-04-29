import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

sys.path.append('resources')
from resources.rnn import RLRNN
from resources.bandits import AgentQ, AgentNetwork, BanditsDrift, BanditsSwitch, plot_session, create_dataset as create_dataset_bandits
from resources.sindy_utils import check_library_setup
from resources.rnn_utils import parameter_file_naming
from resources.sindy_training import fit_spice
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session as plot_session
from utils.setup_agents import setup_agent_rnn

warnings.filterwarnings("ignore")

def main(
    model: str = None,
    data: str = None,
    
    # generated dataset parameters
    participant_id: int = None,
    
    # sindy parameters
    optimizer_threshold = 0.05,
    polynomial_degree = 2,
    optimizer_alpha = 1,
    n_trials_off_policy = 1024,
    n_sessions_off_policy = 0,
    verbose = True,
    
    # ground truth parameters
    beta_reward = 3.,
    alpha = 0.25,
    alpha_penalty = -1.,
    forget_rate = 0.,
    confirmation_bias = 0.,
    beta_choice = 0.,
    alpha_choice = 0.,
    alpha_counterfactual = 0.,
    parameter_variance = 0.,
    
    # environment parameters
    n_actions = 2,
    sigma = .2,
    counterfactual = False,
    
    get_loss: bool = False,
    analysis: bool = False, 
    ):

    # ---------------------------------------------------------------------------------------------------
    # SINDy-agent setup
    # ---------------------------------------------------------------------------------------------------
    
    # tracked variables and control signals in the RNN
    list_rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
    list_control_parameters = ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice']
    sindy_feature_list = list_rnn_modules + list_control_parameters

    # library setup: 
    # which terms are allowed as control inputs in each SINDy model
    # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
    library_setup = {
        'x_learning_rate_reward': ['c_reward', 'c_value_reward', 'c_value_choice'],
        'x_value_reward_not_chosen': ['c_value_choice'],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
    }

    # data-filter setup: 
    # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
    # key is the SINDy model name, value is a list with a triplet of values:
    #   1. str: feature name to be used as a filter
    #   2. numeric: the numeric filter condition
    #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
    # Multiple conditions can also be given as a list of triplets.
    # Example:
    #   'x_value_choice_not_chosen': ['c_action', 0, True] means that for the SINDy model 'x_value_choice_not_chosen', only samples where the feature 'c_action' == 0 are used for training the SINDy model. 
    #   The control parameter 'c_action' is removed afterwards from the list of control signals for training of the model
    filter_setup = {
        # 'x_value_reward_chosen': ['c_action', 1, True], -> Remove this one as well
        'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }

    # data pre-processing setup:
    # define the processing steps for each variable and control signal.
    # possible processing steps are: 
    #   1. Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
    #   2. Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
    #   3. Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
    # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
    dataprocessing_setup = {
        'x_learning_rate_reward': [0, 0, 0],
        'x_value_reward_not_chosen': [0, 0, 0],
        'x_value_choice_chosen': [0, 0, 0],
        'x_value_choice_not_chosen': [0, 0, 0],
        # 'c_action': [0, 0, 0],
        # 'c_reward': [0, 0, 0],
        'c_value_reward': [0, 0, 0],
        'c_value_choice': [0, 0, 0],
    }
    
    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')
        
    # ---------------------------------------------------------------------------------------------------
    # RNN Setup
    # ---------------------------------------------------------------------------------------------------
    
    # set up rnn agent and expose q-values to train sindy
    if model is None:
        params_path = parameter_file_naming('params/params', beta_reward=beta_reward, alpha_reward=alpha, alpha_penalty=alpha_penalty, beta_choice=beta_choice, alpha_choice=alpha_choice, forget_rate=forget_rate, confirmation_bias=confirmation_bias, alpha_counterfactual=alpha_counterfactual, variance=parameter_variance, verbose=True)
    else:
        params_path = model
    
    agent_rnn = setup_agent_rnn(
        path_model=params_path,
        list_sindy_signals=list_rnn_modules+list_control_parameters,
    )
    
    # ---------------------------------------------------------------------------------------------------
    # Data setup
    # ---------------------------------------------------------------------------------------------------
    
    agent = None
    if data is None:
        # set up ground truth agent and environment
        environment = BanditsDrift(sigma=sigma, n_actions=n_actions, counterfactual=counterfactual)
        agent = AgentQ(
            n_actions=n_actions, 
            beta_reward=beta_reward, 
            alpha_reward=alpha, 
            alpha_penalty=alpha_penalty, 
            beta_choice=beta_choice, 
            alpha_choice=alpha_choice, 
            forget_rate=forget_rate, 
            confirmation_bias=confirmation_bias, 
            alpha_counterfactual=alpha_counterfactual,
            )
        
        if participant_id is not None or participant_id != 0:
            participant_id = 0
        participant_ids = np.arange(agent_rnn._model.n_participants, dtype=int)
        dataset_test, _, _ = create_dataset_bandits(agent, environment, 200, len(participant_ids))
    else:
        # get data from experiments for later evaluation
        dataset_test, _, df, _ = convert_dataset(data)
        participant_ids = dataset_test.xs[..., -1].unique().int().cpu().numpy()
        
        # TODO: DELETE AGAIN!!!!!
        # TESTING: bring some noise into sequence length
        # if not (dataset_test.xs.cpu().numpy() == -1).any():
        #     n_trials = dataset_test.xs.shape[1]
        #     for index_session in range(len(dataset_test)):
        #         n_trials_short = int(np.random.uniform(low=0, high=n_trials*0.5))
        #         dataset_test.xs[index_session, -n_trials_short:, :-1] = -1
        #         dataset_test.ys[index_session, -n_trials_short:] = -1
        
        
    # ---------------------------------------------------------------------------------------------------
    # SINDy training
    # ---------------------------------------------------------------------------------------------------
    
    # setup the SINDy-agent
    agent_spice, loss_spice = fit_spice(
        rnn_modules=list_rnn_modules,
        control_signals=list_control_parameters,
        agent=agent_rnn,
        data=dataset_test,
        n_sessions_off_policy=n_sessions_off_policy,
        n_trials_off_policy=n_trials_off_policy,
        polynomial_degree=polynomial_degree,
        library_setup=library_setup,
        filter_setup=filter_setup,
        dataprocessing=dataprocessing_setup,
        optimizer_threshold=optimizer_threshold,
        optimizer_alpha=optimizer_alpha,
        get_loss=get_loss,
        participant_id=participant_id,
        shuffle=True,
        verbose=verbose,
    )

    # ---------------------------------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------------------------------
    
    if analysis:
        
        participant_id_test = participant_id if participant_id is not None else participant_ids[0]
        
        agent_rnn.new_sess(participant_id=participant_id_test)
        agent_spice.new_sess(participant_id=participant_id_test)
        
        # print sindy equations from tested sindy agent
        print('\nDiscovered SPICE models:')
        for model in list_rnn_modules:
            agent_spice._model.submodules_sindy[model][participant_id_test].print()
        print('\n')
        
        # set up ground truth agent by getting parameters from dataset if specified
        if data is not None and agent is None and analysis and 'mean_beta_reward' in df.columns:
            agent = AgentQ(
                beta_reward = df['beta_reward'].values[(df['session']==participant_id_test).values][0],
                alpha_reward = df['alpha_reward'].values[(df['session']==participant_id_test).values][0],
                alpha_penalty = df['alpha_penalty'].values[(df['session']==participant_id_test).values][0],
                confirmation_bias = df['confirmation_bias'].values[(df['session']==participant_id_test).values][0],
                forget_rate = df['forget_rate'].values[(df['session']==participant_id_test).values][0],
                beta_choice = df['beta_choice'].values[(df['session']==participant_id_test).values][0],
                alpha_choice = df['alpha_choice'].values[(df['session']==participant_id_test).values][0],
            )
        
        # get analysis plot
        if agent is not None:
            agents = {'groundtruth': agent, 'rnn': agent_rnn, 'sindy': agent_spice}
            plt_title = r'$GT:\beta_{reward}=$'+str(np.round(agent._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent._beta_choice, 2))+'\n'
        else:
            agents = {'rnn': agent_rnn, 'sindy': agent_spice}
            plt_title = ''
            
        fig, axs = plot_session(agents, dataset_test.xs[participant_id_test])
        betas = agent_spice.get_betas()
        plt_title += r'SINDy: $\beta_{reward}=$'+str(np.round(betas['x_value_reward'], 2)) + r'; $\beta_{choice}=$'+str(np.round(betas['x_value_choice'], 2))
        
        fig.suptitle(plt_title)
        plt.show()
        
    features = {}
    for model in agent_spice._model.submodules_sindy:
        features[model] = {}
        for pid in agent_spice._model.submodules_sindy[model]:
            features_i = np.array(agent_spice._model.submodules_sindy[model][pid].get_feature_names())
            coeffs_i = agent_spice._model.submodules_sindy[model][pid].coefficients()[0]
            index_u = [not 'dummy' in f for f in features_i]
            features_i = features_i[index_u]
            coeffs_i = coeffs_i[index_u]
            # features[model][pid] = tuple(features_i)
            features[model][pid] = tuple(coeffs_i)
    
    features['beta_reward'] = {}
    features['beta_choice'] = {}
    for pid in participant_ids:
        pid = int(pid)
        agent_spice.new_sess(participant_id=pid)
        betas = agent_spice.get_betas()
        features['beta_reward'][pid] = betas['x_value_reward']
        features['beta_choice'][pid] = betas['x_value_choice']
        
    return agent_spice, features, loss_spice


if __name__=='__main__':
    main(
        model = 'params/benchmarking/rnn_sugawara.pkl',
        data = 'data/2arm/sugawara2021_143_processed.csv',
        n_trials=None,
        n_sessions=None,
        verbose=False,
        
        # sindy parameters
        polynomial_degree=2,
        optimizer_threshold=0.05,
        optimizer_alpha=0,

        # generated training dataset parameters
        # n_trials_per_session = 200,
        # n_sessions = 100,
        
        # ground truth parameters
        # alpha = 0.25,
        # beta = 3,
        # forget_rate = 0.,
        # perseverance_bias = 0.25,
        # alpha_penalty = 0.5,
        # confirmation_bias = 0.5,
        # reward_update_rule = lambda q, reward: reward-q,
        
        # environment parameters
        # sigma = 0.1,
        
        analysis=True,
    )