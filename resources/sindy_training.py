from typing import List, Union, Dict, Tuple
import numpy as np
from math import comb
import torch
from copy import deepcopy

import pysindy as ps

from resources.sindy_utils import remove_control_features, conditional_filtering, create_dataset
from resources.rnn_utils import DatasetRNN
from resources.bandits import AgentNetwork, AgentSpice, Bandits, BanditsDrift, AgentQ, create_dataset as create_dataset_bandits


def fit_sindy(
    variables: List[np.ndarray], 
    control: List[np.ndarray] = None, 
    feature_names: List[str] = None, 
    polynomial_degree: int = 1, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    verbose: bool = False,
    get_loss: bool = False,
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 1,
    ):
    
    if feature_names is None:
        if len(library_setup) > 0:
            raise ValueError('If library_setup is provided, feature_names must be provided as well.')
        if len(filter_setup) > 0:
            raise ValueError('If datafilter_setup is provided, feature_names must be provided as well.')
        feature_names = [f'x{i}' for i in range(variables[0].shape[-1])]
    
    # get all x-features
    x_features = [feature for feature in feature_names if feature.startswith('x_')]
    # get all control features
    c_features = [feature for feature in feature_names if feature.startswith('c_')]
    
    # make sure that all x_features are in the library_setup
    for feature in x_features:
        if feature not in library_setup:
            library_setup[feature] = []
    
    # train one sindy model per variable
    sindy_models = {feature: None for feature in x_features}
    loss = 0
    for i, x_feature in enumerate(x_features):
        if verbose:
            print(f'\nSINDy model for {x_feature}:')
        
        # sort signals into corresponding arrays    
        x_i = [x.reshape(-1, 1) for x in variables[:, :, i]]  # get current x-feature as target variable
        x_to_control = variables[:, :, i != np.arange(variables.shape[-1])]  # get all other x-features as control variables
        control_i = [c for c in np.concatenate([x_to_control, control], axis=-1)]  # concatenate control variables with control features
        feature_names_i = [x_feature] + np.array(x_features)[i != np.arange(variables.shape[-1])].tolist() + c_features
        
        # filter target variable and control features according to filter conditions
        if x_feature in filter_setup:
            if not isinstance(filter_setup[x_feature][0], list):
                # check that filter_setup[x_feature] is a list of filter-conditions 
                filter_setup[x_feature] = [filter_setup[x_feature]]
            for filter_condition in filter_setup[x_feature]:
                x_i, control_i, feature_names_i = conditional_filtering(x_i, control_i, feature_names_i, filter_condition[0], filter_condition[1], filter_condition[2])
        
        # remove unnecessary control features according to library setup
        control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
        feature_names_i = [x_feature] + library_setup[x_feature]
        
        # add a dummy control feature if no control features are remaining - otherwise sindy breaks --> TODO: find out why
        if control_i is None or len(control_i) == 0:
            control_i = [np.zeros_like(x_i[0]) for _ in range(len(x_i))]
            feature_names_i = feature_names_i + ['dummy']
        
        # set up increasing thresholds with polynomial degree
        n_polynomial_combinations = np.array([comb(len(feature_names_i) + d, d) for d in range(polynomial_degree+1)])
        thresholds = np.zeros((1, n_polynomial_combinations[-1]))
        index = 0
        for d in range(len(n_polynomial_combinations)):
            thresholds[0, index:n_polynomial_combinations[d]] = d * optimizer_threshold
            index = n_polynomial_combinations[d]
        
        # setup sindy model for current x-feature
        sindy_models[x_feature] = ps.SINDy(
            # optimizer=ps.STLSQ(threshold=optimizer_threshold, alpha=optimizer_alpha, verbose=verbose),
            optimizer=ps.SR3(thresholder="weighted_l1", nu=optimizer_alpha, threshold=optimizer_threshold, thresholds=thresholds, verbose=verbose),
            feature_library=ps.PolynomialLibrary(polynomial_degree, include_bias=True),
            discrete_time=True,
            feature_names=feature_names_i,
        )
        
        # fit sindy model
        sindy_models[x_feature].fit(x_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False)
        
        # post-process sindy weights
        coefs = sindy_models[x_feature].coefficients()
        for index_feature, feature in enumerate(sindy_models[x_feature].get_feature_names()):
            if np.abs(coefs[0, index_feature]) < optimizer_threshold:
                sindy_models[x_feature].model.steps[-1][1].coef_[0, index_feature] = 0.
            if feature == x_feature and np.abs(1-coefs[0, index_feature]) < optimizer_threshold:
                sindy_models[x_feature].model.steps[-1][1].coef_[0, index_feature] = 1.
                
        if get_loss:
            loss_model = 1-sindy_models[x_feature].score(x_i, u=control_i, t=1, multiple_trajectories=True)
            loss += loss_model
            if verbose:
                print(f'Score for {x_feature}: {loss_model}')
        if verbose:
            sindy_models[x_feature].print()
    
    if get_loss:
        return sindy_models, loss
    else:
        return sindy_models
    
    
def fit_model(
    rnn_modules: List[np.ndarray],
    control_parameters: List[np.ndarray], 
    agent: AgentNetwork,
    data: DatasetRNN,
    polynomial_degree: int = 2, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 1,
    participant_id: int = None,
    shuffle: bool = False,
    dataprocessing: Dict[str, List] = None,
    off_policy: bool = True,
    n_trials_off_policy: int = 1024,
    deterministic: bool = True,
    # get_loss: bool = False,
    verbose: bool = False,
    ) -> AgentSpice:
    
    if participant_id is not None:
        participant_ids = [participant_id]
        data = DatasetRNN(xs=data.xs[participant_id][None], ys=data.ys[participant_id][None])
    else:
        participant_ids = data.xs[..., -1].unique().int().cpu().numpy()
    
    if off_policy:
        # set up environment to create an off-policy dataset (w.r.t to trained RNN) of arbitrary length
        # The trained RNN will then perform value updates to get off-policy data
        environment = BanditsDrift(sigma=0.2, n_actions=agent._n_actions)
        agent_dummy = AgentQ(n_actions=agent._n_actions, alpha_reward=0.5, beta_reward=1.0)
        dataset_off_policy = create_dataset_bandits(agent=agent_dummy, environment=environment, n_trials=n_trials_off_policy, n_sessions=1)[0]
        # repeat the off-policy data for every participant and add the corresponding participant ID
        xs_off_policy, ys_off_policy = torch.zeros((data.xs.shape[0], n_trials_off_policy, data.xs.shape[-1])), torch.zeros((data.ys.shape[0], n_trials_off_policy, data.ys.shape[-1]))
        for index_data in range(len(data)):
            xs_off_policy[index_data] = dataset_off_policy.xs
            xs_off_policy[index_data, :, -1] = data.xs[index_data, :1, -1].repeat(n_trials_off_policy)
            ys_off_policy[index_data] = dataset_off_policy.ys
        data = DatasetRNN(xs=xs_off_policy, ys=ys_off_policy)
    
    sindy_models = {rnn_module: {} for rnn_module in rnn_modules}
    for participant_id in participant_ids:
        # extract all necessary data from the RNN (memory state) and align with the control inputs (action, reward)
        mask_participant = data.xs[:, 0, -1] == participant_id
        variables, control_parameters, feature_names, _ = create_dataset(
            agent=agent,
            data=DatasetRNN(*data[mask_participant]),
            participant_id=participant_id,
            shuffle=shuffle,
            dataprocessing=dataprocessing,
        )

        # fit one SINDy-model per RNN-module
        sindy_models_id = fit_sindy(
            variables=variables,
            control=control_parameters,
            feature_names=feature_names,
            polynomial_degree=polynomial_degree,
            library_setup=library_setup,
            filter_setup=filter_setup,
            optimizer_alpha=optimizer_alpha,
            optimizer_threshold=optimizer_threshold,
            verbose=verbose,
        )
        
        for rnn_module in rnn_modules:
            sindy_models[rnn_module][participant_id] = sindy_models_id[rnn_module]

    # set up a SINDy-based agent by replacing the RNN-modules with the respective SINDy-model
    agent_spice = AgentSpice(model_rnn=deepcopy(agent._model), sindy_modules=sindy_models, n_actions=agent._n_actions, deterministic=deterministic)
    
    return agent_spice