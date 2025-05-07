import optuna
import numpy as np
import pysindy as ps
from math import comb
from copy import deepcopy

def optimize_for_participant(
    variables, 
    control, 
    rnn_modules, 
    control_signals, 
    library_setup, 
    filter_setup,
    polynomial_degree=1,
    n_trials=50,
    timeout=600,  # ? 10 minutes timeout
    verbose=False
):
    """
    Use Optuna to find the best optimizer type and hyperparameters for a specific participant.
    
    Args:
        variables: RNN module variables for the participant
        control: Control signals for the participant
        rnn_modules: List of RNN module names
        control_signals: List of control signal names
        library_setup: Dictionary mapping features to library components
        filter_setup: Dictionary mapping features to filter conditions
        polynomial_degree: Polynomial degree for the feature library
        n_trials: Number of Optuna trials to run
        timeout: Maximum time in seconds to run optimization
        verbose: Whether to print verbose output
        
    Returns:
        dict: Best optimizer configuration with type and parameters
    """
    from resources.sindy_utils import remove_control_features, conditional_filtering

    def objective(trial):
        # Sample optimizer type
        optimizer_type = trial.suggest_categorical(
            "optimizer_type", ["STLSQ", "SR3_L1", "SR3_weighted_l1"]
        )
        
        # Sample hyperparameters
        optimizer_alpha = trial.suggest_float("optimizer_alpha", 0.01, 1.0, log=True)
        optimizer_threshold = trial.suggest_float("optimizer_threshold", 0.01, 0.2, log=True)
        
        # Initialize prediction error
        total_error = 0
        
        # For each RNN module, fit a SINDy model and compute prediction error
        for index_feature, x_feature in enumerate(rnn_modules):
            # Extract data for this module
            x_i = [x[:, index_feature].reshape(-1, 1) for x in variables]
            control_i = control
            feature_names_i = [x_feature] + control_signals
            
            # Apply filters if specified
            if x_feature in filter_setup:
                if not isinstance(filter_setup[x_feature][0], list):
                    filter_setup[x_feature] = [filter_setup[x_feature]]
                for filter_condition in filter_setup[x_feature]:
                    x_i, control_i, feature_names_i = conditional_filtering(
                        x_train=x_i, 
                        control=control_i, 
                        feature_names=feature_names_i, 
                        feature_filter=filter_condition[0], 
                        condition=filter_condition[1], 
                        remove_feature_filter=False
                    )
            
            # Remove unnecessary control features
            control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
            feature_names_i = [x_feature] + library_setup[x_feature]
            
            # Handle case with no control features
            if control_i is None or len(control_i) == 0:
                try:
                    # Add dummy control
                    control_i = [np.zeros_like(x_i[0]) for _ in range(len(x_i))]
                    feature_names_i = feature_names_i + ['dummy']
                except:
                    # If this fails, return a high error
                    return float('inf')
            
            # Set up thresholds for weighted SR3
            if optimizer_type == "SR3_weighted_l1":
                n_polynomial_combinations = np.array([comb(len(feature_names_i) + d, d) for d in range(polynomial_degree+1)])
                thresholds = np.zeros((1, n_polynomial_combinations[-1]))
                index = 0
                for d in range(len(n_polynomial_combinations)):
                    thresholds[0, index:n_polynomial_combinations[d]] = d * optimizer_threshold
                    index = n_polynomial_combinations[d]
            
            # Create optimizer based on type
            if optimizer_type == "STLSQ":
                optimizer = ps.STLSQ(alpha=optimizer_alpha, threshold=optimizer_threshold)
            elif optimizer_type == "SR3_L1":
                optimizer = ps.SR3(
                    thresholder="L1",
                    nu=optimizer_alpha,
                    threshold=optimizer_threshold,
                    max_iter=100
                )
            else:  # "SR3_weighted_l1"
                optimizer = ps.SR3(
                    thresholder="weighted_l1",
                    nu=optimizer_alpha,
                    threshold=optimizer_threshold,
                    thresholds=thresholds,
                    max_iter=100
                )
            
            # Create and fit SINDy model
            try:
                model = ps.SINDy(
                    optimizer=optimizer,
                    feature_library=ps.PolynomialLibrary(polynomial_degree),
                    discrete_time=True,
                    feature_names=feature_names_i
                )
                
                model.fit(x_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False)
                
                # Calculate prediction error using validation approach
                error = 0
                for j in range(len(x_i)):
                    # Use first 80% for training, last 20% for validation
                    train_len = int(0.8 * len(x_i[j]))
                    if train_len < 10:  # Skip if too few samples
                        continue
                        
                    x_train = x_i[j][:train_len]
                    x_val = x_i[j][train_len:-1]  # -1 because we need x_val+1 for comparison
                    
                    if len(control_i[j].shape) > 1 and control_i[j].shape[1] > 0:
                        u_train = control_i[j][:train_len]
                        u_val = control_i[j][train_len:-1]
                        
                        # Retrain on this trajectory's training data
                        model_traj = deepcopy(model)
                        model_traj.fit(x_train.reshape(-1, 1), u=u_train, t=1)
                        
                        # Predict on validation data
                        x_pred = model_traj.predict(x_val, u=u_val)
                    else:
                        continue  # Skip if no control signals
                    
                    # Compare predictions with actual next states
                    x_next = x_i[j][train_len+1:]
                    mse = np.mean((x_next - x_pred) ** 2)
                    error += mse
                
                # Add complexity penalty (AIC-like)
                n_nonzero_coeffs = np.count_nonzero(model.coefficients())
                complexity_penalty = 0.01 * n_nonzero_coeffs
                
                # Add to total error
                total_error += error + complexity_penalty
                
            except Exception as e:
                if verbose:
                    print(f"Error fitting model for {x_feature}: {str(e)}")
                return float('inf')
        
        return total_error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Extract best parameters
    best_params = study.best_params
    best_optimizer_type = best_params.pop("optimizer_type")
    
    if verbose:
        print(f"Best optimizer type: {best_optimizer_type}")
        print(f"Best parameters: alpha={best_params['optimizer_alpha']}, threshold={best_params['optimizer_threshold']}")
        print(f"Best objective value: {study.best_value}")
    
    return {
        "optimizer_type": best_optimizer_type,
        "optimizer_alpha": best_params["optimizer_alpha"],
        "optimizer_threshold": best_params["optimizer_threshold"],
    }