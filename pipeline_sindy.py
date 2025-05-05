import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional

sys.path.append('resources')
from resources.rnn import RLRNN
from resources.bandits import AgentQ, AgentNetwork, BanditsDrift, BanditsSwitch, plot_session, create_dataset as create_dataset_bandits, get_update_dynamics
from resources.sindy_utils import check_library_setup
from resources.rnn_utils import parameter_file_naming, DatasetRNN
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
    optimizer_type: str = "SR3_L1",
    optimizer_threshold = 0.05,
    polynomial_degree = 2,
    optimizer_alpha = 1,
    n_trials_off_policy = 1024,
    n_sessions_off_policy = 0,
    verbose = True,
    use_optuna = False,
    filter_bad_participants = False,  # Added parameter to control filtering
    
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
    show_plots: bool = True,  # Added parameter to control plot display
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
        'x_value_choice_chosen': [1, 0, 1],
        'x_value_choice_not_chosen': [1, 0, 1],
        # 'c_action': [0, 0, 0],
        # 'c_reward': [0, 0, 0],
        'c_value_reward': [0, 0, 0],
        'c_value_choice': [1, 0, 1],
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
        
    # ---------------------------------------------------------------------------------------------------
    # SINDy training
    # ---------------------------------------------------------------------------------------------------
    
    if use_optuna and verbose:
        print("\nUsing Optuna to find optimal optimizer configuration for each participant")
    
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
        optimizer_type=optimizer_type,
        optimizer_threshold=optimizer_threshold,
        optimizer_alpha=optimizer_alpha,
        get_loss=get_loss,
        participant_id=participant_id,
        shuffle=True,
        verbose=verbose,
        use_optuna=use_optuna,
    )

    # Filter out badly fitted SPICE models
    filtered_participant_ids = []
    if filter_bad_participants and agent_spice is not None and dataset_test is not None:
        if verbose:
            print("\nFiltering badly fitted SPICE models...")
            
        # Create a copy of the valid participant IDs
        valid_participant_ids = list(participant_ids)
        
        for index_session, pid in enumerate(participant_ids):
            # Skip if participant is not in the SPICE model
            if pid not in agent_spice._model.submodules_sindy[list_rnn_modules[0]]:
                continue
                
            # Calculate normalized log likelihood for SPICE and RNN
            mask_participant_id = dataset_test.xs[:, 0, -1] == pid
            if not mask_participant_id.any():
                continue
                
            participant_data = DatasetRNN(*dataset_test[mask_participant_id])
            
            # Get predictions from both models
            n_trials_test = len(participant_data.xs[0])
            
            # Get probabilities from SPICE and RNN models
            agent_spice.new_sess(participant_id=pid)
            agent_rnn.new_sess(participant_id=pid)
            
            # Get experiment data
            experiment = participant_data.xs.cpu().numpy()[0]
            
            try:
                # Calculate NLL for both models
                result_spice = get_update_dynamics(experiment=experiment, agent=agent_spice)
                result_rnn = get_update_dynamics(experiment=experiment, agent=agent_rnn)
                probs_spice = result_spice[1]  # Get the second returned value (choice_probs)
                probs_rnn = result_rnn[1]  # Get the second returned value (choice_probs)   
                          
              # Get the probabilities for the chosen actions      
                # Calculate scores (negative log likelihood)
                scores_spice = -np.sum(np.log(np.clip(probs_spice, 1e-10, 1.0)))
                scores_rnn = -np.sum(np.log(np.clip(probs_rnn, 1e-10, 1.0)))
                
                # Check if SPICE model is badly fitted
                spice_per_action_likelihood = np.exp(-scores_spice/(n_trials_test*agent_rnn._n_actions))
                rnn_per_action_likelihood = np.exp(-scores_rnn/(n_trials_test*agent_rnn._n_actions))
                
                if spice_per_action_likelihood < 0.6 and rnn_per_action_likelihood > 0.63:
                    if verbose:
                        print(f'SPICE NLL ({scores_spice:.2f}) is unplausibly high compared to RNN NLL ({scores_rnn:.2f}).')
                        print(f'SPICE fitting seems badly parameterized. Skipping participant {pid}.')
                    
                    # Remove this participant from the SPICE model
                    for module in agent_spice._model.submodules_sindy:
                        if pid in agent_spice._model.submodules_sindy[module]:
                            del agent_spice._model.submodules_sindy[module][pid]
                    
                    # Remove from valid participant IDs
                    if pid in valid_participant_ids:
                        valid_participant_ids.remove(pid)
                else:
                    # Keep track of filtered (good) participants
                    filtered_participant_ids.append(pid)
            except Exception as e:
                if verbose:
                    print(f"Error evaluating participant {pid}: {str(e)}")
                if pid in valid_participant_ids:
                    valid_participant_ids.remove(pid)
        
        # Update participant_ids to only include valid ones
        participant_ids = np.array(valid_participant_ids)
        
        if verbose:
            print(f"\nAfter filtering: {len(filtered_participant_ids)} of {len(filtered_participant_ids) + len(participant_ids) - len(filtered_participant_ids)} participants have well-fitted SPICE models.")

    # ---------------------------------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------------------------------
    
    if analysis and len(participant_ids) > 0:
        participant_id_test = participant_id if participant_id is not None else participant_ids[0]
        
        agent_rnn.new_sess(participant_id=participant_id_test)
        agent_spice.new_sess(participant_id=participant_id_test)
        
        # print sindy equations from tested sindy agent
        print('\nDiscovered SPICE models:')
        for model in list_rnn_modules:
            if participant_id_test in agent_spice._model.submodules_sindy[model]:
                agent_spice._model.submodules_sindy[model][participant_id_test].print()
            else:
                print(f"No model available for {model}, participant {participant_id_test}")
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
        
        # Only show plots if show_plots is True
        if show_plots:
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
    
    # If agent_spice is None, we couldn't fit the model, so return early
    if agent_spice is None:
        print("ERROR: Failed to fit SPICE model. Returning None.")
        return None, None, None
        
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
        if pid in agent_spice._model.submodules_sindy[list_rnn_modules[0]]:
            agent_spice.new_sess(participant_id=pid)
            betas = agent_spice.get_betas()
            features['beta_reward'][pid] = betas['x_value_reward']
            features['beta_choice'][pid] = betas['x_value_choice']
        
    return agent_spice, features, loss_spice


if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SINDy pipeline with optimizer selection")
    parser.add_argument("--model", type=str, default='params/benchmarking/rnn_sugawara.pkl',
                        help="Path to model file")
    parser.add_argument("--data", type=str, default='data/2arm/sugawara2021_143_processed.csv',
                        help="Path to data file")
    parser.add_argument("--participant_id", type=int, default=None, 
                        help="Participant ID")
    parser.add_argument("--polynomial_degree", type=int, default=2, 
                        help="Polynomial degree")
    parser.add_argument("--optimizer_type", type=str, default="SR3_L1", 
                        choices=["STLSQ", "SR3_L1", "SR3_weighted_l1"],
                        help="Optimizer type")
    parser.add_argument("--optimizer_alpha", type=float, default=0.1, 
                        help="Optimizer alpha")
    parser.add_argument("--optimizer_threshold", type=float, default=0.05, 
                        help="Optimizer threshold")
    parser.add_argument("--use_optuna", action="store_true", 
                        help="Use Optuna to find the best optimizer for each participant")
    parser.add_argument("--filter_bad_participants", action="store_true", 
                        help="Filter out badly fitted participants")
    parser.add_argument("--n_trials_off_policy", type=int, default=1024, 
                        help="Number of trials for off-policy data")
    parser.add_argument("--n_sessions_off_policy", type=int, default=0, 
                        help="Number of sessions for off-policy data")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose output")
    parser.add_argument("--analysis", action="store_true", 
                        help="Perform analysis")
    parser.add_argument("--get_loss", action="store_true", 
                        help="Compute loss")
    parser.add_argument("--show_plots", action="store_true", 
                        help="Show analysis plots")
    
    args = parser.parse_args()
    
    agent_spice, features, loss = main(
        model=args.model,
        data=args.data,
        participant_id=args.participant_id,
        polynomial_degree=args.polynomial_degree,
        optimizer_type=args.optimizer_type,
        optimizer_alpha=args.optimizer_alpha,
        optimizer_threshold=args.optimizer_threshold,
        n_trials_off_policy=args.n_trials_off_policy,
        n_sessions_off_policy=args.n_sessions_off_policy,
        verbose=args.verbose,
        analysis=args.analysis,
        get_loss=args.get_loss,
        use_optuna=args.use_optuna,
        filter_bad_participants=args.filter_bad_participants,
        show_plots=args.show_plots,
    )
    
    if loss is not None:
        print(f"Loss: {loss}")