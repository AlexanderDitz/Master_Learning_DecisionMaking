from typing import List, Union, Dict, Tuple
import numpy as np
from math import comb
import torch
from copy import deepcopy

import pysindy as ps

from resources.sindy_utils import remove_control_features, conditional_filtering, create_dataset
from resources.rnn_utils import DatasetRNN
from resources.bandits import AgentNetwork, AgentSindy, Bandits


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
        coefs = sindy_models[x_feature].model.steps[-1][1].coef_
        for index_feature, feature in enumerate(sindy_models[x_feature].get_feature_names()):
            # # case: coefficient is x_feature[k] 
            # # --> Target in the case of non-available dynamics: 
            # # x_feature[k+1] = 1.0 x_feature[k] and not e.g. x_feature[k+1] = 1.03 x_feature[k]
            # if feature == x_feature:
            #     if np.abs(coefs[0, index_feature]-1) < optimizer_threshold:
            #         sindy_models[x_feature].model.steps[-1][1].coef_[0, index_feature] = 1.
            #     elif np.abs(coefs[0, index_feature]) < optimizer_threshold:
            #         sindy_models[x_feature].model.steps[-1][1].coef_[0, index_feature] = 0.
            # # case: any other coefficient
            # elif np.abs(coefs[0, index_feature]) < optimizer_threshold:
            #     sindy_models[x_feature].model.steps[-1][1].coef_[0, index_feature] = 0.
            if np.abs(coefs[0, index_feature]) < optimizer_threshold:
                sindy_models[x_feature].model.steps[-1][1].coef_[0, index_feature] = 0.
        
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
    data: Union[Bandits, DatasetRNN],
    polynomial_degree: int = 2, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 1,
    n_trials: int = 1024,
    n_sessions: int = 1,
    participant_id: int = None,
    shuffle: bool = False,
    dataprocessing: Dict[str, List] = None,
    # get_loss: bool = False,
    verbose: bool = False,
    ) -> AgentSindy:
    
    if participant_id is not None:
        if isinstance(participant_id, int):
            participant_ids = [participant_id]
        else:
            raise TypeError(f'The argument participant_id must be of type None or Integer.')
    else:
        if isinstance(data, DatasetRNN):
            participant_ids = data.xs[..., -1].unique()
        elif isinstance(data, Bandits):
            participant_ids = np.arange(n_sessions)
        else:
            raise TypeError(f'The argument data must be of type (numpy.ndarray, torch.Tensor, DatasetRNN).')
    n_participants = len(participant_ids)
    
    sindy_models = {rnn_module: {} for rnn_module in rnn_modules}
    for participant_id in range(n_participants):
        # extract all necessary data from the RNN (memory state) and align with the control inputs (action, reward)
        variables, control_parameters, feature_names, _ = create_dataset(
            agent=agent,
            data=data,
            n_trials=n_trials,
            n_sessions=n_sessions,
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
    agent_sindy = AgentSindy(model_rnn=deepcopy(agent._model), sindy_modules=sindy_models, n_actions=agent._n_actions)
    
    return agent_sindy