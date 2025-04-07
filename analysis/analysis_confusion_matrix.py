import sys, os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_spice, setup_agent_mcmc
from resources.bandits import get_update_dynamics, AgentSpice
from resources.model_evaluation import log_likelihood, bayesian_information_criterion, akaike_information_criterion

# Simulation settings
models = ["Baseline", "Benchmark"]#, "Spice"]
n_actions = 2

# # Use testing data to avoid overfitting
# path_data = {
#     "Baseline": "data/sugawara2021/sugawara2021_testing_ApBr.csv",
#     "Benchmark": "data/sugawara2021/sugawara2021_testing_ApAnAcBcBr.csv",
#     "Spice": "data/sugawara2021/sugawara2021_testing_Spice.csv",
# }

# path_model = {
#     "Baseline": {
#         "Baseline": "params/sugawara2021/params_sugawara2021_ApBr_ApBr.nc",
#         "Benchmark": "params/sugawara2021/params_sugawara2021_ApAnAcBcBr_ApBr.nc",
#         "Spice": "params/sugawara2021/params_sugawara2021_Spice_ApBr.nc",
#         },
#     "Benchmark": {
#         "Baseline": "params/sugawara2021/params_sugawara2021_ApBr_ApAnAcBcBr.nc",
#         "Benchmark": "params/sugawara2021/params_sugawara2021_ApAnAcBcBr_ApAnAcBcBr.nc",
#         "Spice": "params/sugawara2021/params_sugawara2021_Spice_ApAnAcBcBr.nc",    
#         },
#     "Spice": {
#         "Baseline": "params/sugawara2021/params_sugawara2021_ApBr_Spice.pkl",
#         "Benchmark": "params/sugawara2021/params_sugawara2021_ApAnAcBcBr_Spice.pkl",
#         "Spice": "params/sugawara2021/params_sugawara2021_Spice_Spice.pkl",
#         },
#     }

# Use testing data to avoid overfitting
path_data = {
    "Baseline": "data/parameter_recovery/data_128p_0_ApBr.csv",
    "Benchmark": "data/parameter_recovery/data_128p_0_ApAnBcBr.csv",
    # "Spice": "data/sugawara2021/sugawara2021_testing_Spice.csv",
}

path_model = {
    "Baseline": {
        "Baseline": "params/parameter_recovery/params_128p_0_ApBr_ApBr.nc",
        "Benchmark": "params/parameter_recovery/params_128p_0_ApAnBcBr_ApBr.nc",
        # "Spice": "params/sugawara2021/params_sugawara2021_Spice_ApBr.nc",
        },
    "Benchmark": {
        "Baseline": "params/parameter_recovery/params_128p_0_ApBr_ApAnBcBr.nc",
        "Benchmark": "params/parameter_recovery/params_128p_0_ApAnBcBr_ApAnBcBr.nc",
        # "Spice": "params/sugawara2021/params_sugawara2021_Spice_ApAnAcBcBr.nc",    
        },
    # "Spice": {
    #     "Baseline": "params/sugawara2021/params_sugawara2021_ApBr_Spice.pkl",
    #     "Benchmark": "params/sugawara2021/params_sugawara2021_ApAnAcBcBr_Spice.pkl",
        # "Spice": "params/sugawara2021/params_sugawara2021_Spice_Spice.pkl",
        # },
    }

# SINDy configuration
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

confusion_matrix_nll = np.zeros((len(models), len(models)))
confusion_matrix_bic = np.zeros((len(models), len(models)))
confusion_matrix_aic = np.zeros((len(models), len(models)))
for index_simulated_model, simulated_model in enumerate(models):
    # get data and choice probabilities from simulated model
    dataset, experiment_list, _, values = convert_dataset(file=path_data[simulated_model])
    n_sessions = len(dataset)
    nll = np.zeros((n_sessions, len(models)))
    bic = np.zeros((n_sessions, len(models)))
    aic = np.zeros((n_sessions, len(models)))
    for index_fitted_model, fitted_model in enumerate(models):
        print(f"Comparing fitted model {fitted_model} to simulated data from {simulated_model}...")
        # agent setup for fitted model
        if fitted_model in ["Baseline", "Benchmark"]:
            agent = setup_agent_mcmc(path_model=path_model[fitted_model][simulated_model])
            n_parameters = 2 if fitted_model == "Baseline" else 5
        else:
            agent = setup_agent_spice(
                path_model=path_model[fitted_model][simulated_model],
                path_data=path_data[simulated_model],
                rnn_modules=rnn_modules, 
                control_parameters=control_parameters, 
                sindy_library_polynomial_degree=sindy_library_polynomial_degree,
                sindy_library_setup=sindy_library_setup,
                sindy_filter_setup=sindy_filter_setup,
                sindy_dataprocessing=sindy_dataprocessing,
                )
            n_parameters = agent.count_parameters(mapping_modules_values={'x_learning_rate_reward': 'x_value_reward', 'x_value_reward_not_chosen': 'x_value_reward', 'x_value_choice_chosen': 'x_value_choice', 'x_value_choice_not_chosen': 'x_value_choice'})
        
        for session in tqdm(range(n_sessions)):
            # get choice probabilities from agent for data from simulated model
            n_trials_experiment = len(experiment_list[session].choices)
            experiment = dataset.xs[session, :n_trials_experiment]
            # choice_probs_simulated = values[0][session, :n_trials_experiment]
            choices = dataset.xs[session, :n_trials_experiment, :n_actions].cpu().numpy()
            choice_probs_fitted = get_update_dynamics(experiment=experiment, agent=agent if isinstance(agent, AgentSpice) else agent[session])[1]

            # compute the averaged trial likelihood for the fitted model predicting the simulated data
            # metrics[session, index_fitted_model] = - log_likelihood(choice_probs_simulated, choice_probs_fitted)
            ll = log_likelihood(choices, choice_probs_fitted)
            n_params = n_parameters[session] if isinstance(agent, AgentSpice) else n_parameters
            nll[session, index_fitted_model] = -ll
            bic[session, index_fitted_model] = bayesian_information_criterion(choices, choice_probs_fitted, n_params, ll=ll)
            aic[session, index_fitted_model] = akaike_information_criterion(choices, choice_probs_fitted, n_params, ll=ll)
    
    # get "best model"-counts for each model for each session
    best_model = np.argmin(nll, axis=-1)
    unique, counts = np.unique(best_model, return_counts=True)
    # Ensure all models have a count in the dictionary
    counts_dict = {model_index: 0 for model_index in range(len(models))}
    counts_dict.update(dict(zip(unique, counts / n_sessions)))

    confusion_matrix_nll[index_simulated_model] = np.array([counts_dict[key] for key in counts_dict])
    
    # get "best model"-counts for each model for each session
    best_model = np.argmin(bic, axis=-1)
    unique, counts = np.unique(best_model, return_counts=True)
    # Ensure all models have a count in the dictionary
    counts_dict = {model_index: 0 for model_index in range(len(models))}
    counts_dict.update(dict(zip(unique, counts / n_sessions)))
    
    confusion_matrix_bic[index_simulated_model] = np.array([counts_dict[key] for key in counts_dict])
    
    # get "best model"-counts for each model for each session
    best_model = np.argmin(aic, axis=-1)
    unique, counts = np.unique(best_model, return_counts=True)
    # Ensure all models have a count in the dictionary
    counts_dict = {model_index: 0 for model_index in range(len(models))}
    counts_dict.update(dict(zip(unique, counts / n_sessions)))
    
    confusion_matrix_aic[index_simulated_model] = np.array([counts_dict[key] for key in counts_dict])
    
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_nll, annot=True, xticklabels=models, yticklabels=models, cmap="viridis", vmax=1, vmin=0)
plt.xlabel("Fitted Model")
plt.ylabel("Simulated Model")
plt.title("Confusion Matrix: p(Fitted Model | Simulated Model); NLL")
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_bic, annot=True, xticklabels=models, yticklabels=models, cmap="viridis", vmax=1, vmin=0)
plt.xlabel("Fitted Model")
plt.ylabel("Simulated Model")
plt.title("Confusion Matrix: p(Fitted Model | Simulated Model); BIC")
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_aic, annot=True, xticklabels=models, yticklabels=models, cmap="viridis", vmax=1, vmin=0)
plt.xlabel("Fitted Model")
plt.ylabel("Simulated Model")
plt.title("Confusion Matrix: p(Fitted Model | Simulated Model); AIC")
plt.show()