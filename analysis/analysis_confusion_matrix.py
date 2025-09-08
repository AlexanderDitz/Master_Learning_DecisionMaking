import sys, os

import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_spice, setup_agent_rnn
from resources.bandits import get_update_dynamics, AgentSpice, AgentNetwork
from resources.model_evaluation import log_likelihood, bayesian_information_criterion, akaike_information_criterion
from resources.sindy_utils import SindyConfig_eckstein2022 as SindyConfig
from resources.rnn import RLRNN_eckstein2022 as class_rnn
from utils.colormap import truncate_colormap
from benchmarking import benchmarking_eckstein2022, benchmarking_dezfouli2019


#----------------------------------------------------------------------------------------------
# CONFIGURATION CONFUSION MATRIX FILES
#----------------------------------------------------------------------------------------------

fitted_models = ["baseline", "benchmark", "spice"]
simulated_models = ["baseline", "benchmark", "rnn"]

study = "eckstein2022"
model_mapping = {"baseline": "ApBr", "benchmark": "ApAnBrBcfBch"}
path_model_benchmark = f'params/{study}/mcmc_{study}_sim_SIMULATED_fit_FITTED.nc'
path_data = f'data/{study}/{study}_generated_behavior_SIMULATED_test.csv'
setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
rl_model = benchmarking_eckstein2022.rl_model

# study = "dezfouli2019"
# model_mapping = {"baseline": "PhiBeta", "benchmark": "PhiChiBetaKappaC"}
# path_model_benchmark = f'params/{study}/gql_{study}_sim_SIMULATED_fit_FITTED.pkl'
# path_data = f'data/{study}/{study}_generated_behavior_SIMULATED_test.csv'
# setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
# Dezfouli2019GQL = benchmarking_dezfouli2019.Dezfouli2019GQL

# file settings
path_model_spice = f'params/{study}/rnn_{study}_SIMULATED.pkl'

n_actions = 2

#----------------------------------------------------------------------------------------------
# SETUP MODELS
#----------------------------------------------------------------------------------------------

print("Simulated models:", simulated_models)
print("Fitted models:", fitted_models)

agents = {}
for simulated_model in simulated_models:
    agents[simulated_model] = {}
    for fitted_model in fitted_models:
        agents[simulated_model][fitted_model] = None
        
        if not fitted_model.lower() in ['rnn', 'spice']:
            agent, n_params = setup_agent_benchmark(
                path_model=path_model_benchmark.replace('SIMULATED', simulated_model).replace('FITTED', fitted_model),
                model_config=model_mapping[fitted_model]
                )
        else:
            if "rnn" in fitted_models:
                agent = setup_agent_rnn(
                    class_rnn=class_rnn, 
                    path_rnn=path_model_spice.replace('SIMULATED', simulated_model), 
                    )
                n_params = 12.647059  # avg n params of eckstein2022-SPICE models
            elif "spice" in fitted_models:
                agent = setup_agent_spice(
                    class_rnn=class_rnn,
                    path_rnn=path_model_spice.replace('SIMULATED', simulated_model),
                    path_spice=path_model_spice.replace('SIMULATED', simulated_model).replace('rnn', 'spice', 1),
                )
                agent.new_sess()
                n_params = agent.count_parameters()
        agents[simulated_model][fitted_model] = (agent, n_params)


#----------------------------------------------------------------------------------------------
# PIPELINE CONFUSION MATRIX
#----------------------------------------------------------------------------------------------

metrics = ['nll', 'aic', 'bic']
confusion_matrix = {metric: np.zeros((len(simulated_models), len(fitted_models))) for metric in metrics}

for index_simulated_model, simulated_model in enumerate(simulated_models):
    
    # get data and choice probabilities from simulated model
    dataset_training = convert_dataset(file=path_data.replace('_test', '_training').replace('SIMULATED', simulated_model))[0]
    dataset_test = convert_dataset(file=path_data.replace('SIMULATED', simulated_model))[0]
    n_sessions = len(dataset_test)
    metrics_session = {metric: np.zeros((n_sessions, len(fitted_models))) for metric in metrics}

    for index_fitted_model, fitted_model in enumerate(fitted_models):
        
        print(f"Comparing fitted model {fitted_model} to simulated data from {simulated_model}...")
        
        # agent setup for fitted model
        agent, n_parameters = agents[simulated_model][fitted_model]
        
        for session in tqdm(range(n_sessions)):
            
            # get choice probabilities from agent for data from simulated model
            choice_probs_fitted_training = get_update_dynamics(experiment=dataset_training.xs[session], agent=agent[session] if isinstance(agent, list) else agent)[1]
            choice_probs_fitted_test = get_update_dynamics(experiment=dataset_test.xs[session], agent=agent[session] if isinstance(agent, list) else agent)[1]
            
            choices_training = dataset_training.xs[session, :len(choice_probs_fitted_training), :n_actions].cpu().numpy()
            choices_test = dataset_test.xs[session, :len(choice_probs_fitted_test), :n_actions].cpu().numpy()
            
            ll_training = log_likelihood(choices_training, choice_probs_fitted_training)
            ll_test = log_likelihood(choices_test, choice_probs_fitted_test)
            
            metrics_session['nll'][session, index_fitted_model] = -ll_test
            metrics_session['bic'][session, index_fitted_model] = bayesian_information_criterion(choices_training, choice_probs_fitted_training, n_parameters[session] if isinstance(n_parameters, list) or isinstance(n_parameters, dict) else n_parameters, ll=ll_training)
            metrics_session['aic'][session, index_fitted_model] = akaike_information_criterion(choices_training, choice_probs_fitted_training, n_parameters[session] if isinstance(n_parameters, list) or isinstance(n_parameters, dict) else n_parameters, ll=ll_training)

    for metric in metrics:
        
        # get "best model"-counts for each model for each session
        best_model = np.argmin(metrics_session[metric], axis=-1)
        unique, counts = np.unique(best_model, return_counts=True)
        
        # Ensure all models have a count in the dictionary
        counts_dict = {model_index: 0 for model_index in range(len(fitted_models))}
        counts_dict.update(dict(zip(unique, counts / n_sessions)))
        
        confusion_matrix[metric][index_simulated_model] += np.array([counts_dict[key] for key in counts_dict])
    
# Plot confusion matrix

def plot_confusion_matrices(confusion_matrix, models, metrics, study, cmap):
    """Plot and save confusion matrices with proper figure management"""
    
    # Create annotated version
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), constrained_layout=True)
    
    for index_metric, metric in enumerate(metrics):
        sns.heatmap(
            confusion_matrix[metric], 
            annot=True, 
            xticklabels=models, 
            yticklabels=models, 
            cmap=cmap,
            vmax=1, 
            vmin=0, 
            ax=axes[index_metric],
            cbar=True,
            fmt='.3f'  # Format annotation numbers
        )
        axes[index_metric].set_xlabel("Fitted Model")
        axes[index_metric].set_ylabel("Simulated Model")
        axes[index_metric].set_title("Confusion Matrix: " + metric.upper())
    
    # Save annotated version
    plt.savefig(f'analysis/plots_confusion_matrix/confusion_matrix_{study}_annotated.png', 
                dpi=500, bbox_inches='tight')
    plt.close(fig)  # Close the figure explicitly
    
    # Create non-annotated version
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), constrained_layout=True)
    
    for index_metric, metric in enumerate(metrics):
        sns.heatmap(
            confusion_matrix[metric], 
            annot=False, 
            xticklabels=models, 
            yticklabels=models, 
            cmap=cmap,
            vmax=1, 
            vmin=0, 
            ax=axes[index_metric],
            cbar=True
        )
        axes[index_metric].set_xlabel("Fitted Model")
        axes[index_metric].set_ylabel("Simulated Model")
        axes[index_metric].set_title("Confusion Matrix: " + metric.upper())
    
    # Save non-annotated version
    plt.savefig(f'analysis/plots_confusion_matrix/confusion_matrix_{study}_not_annotated.png', 
                dpi=500, bbox_inches='tight')
    plt.close(fig)  # Close the figure explicitly

# Call the function
cmap = truncate_colormap(plt.cm.viridis, minval=0.5, maxval=1.0)
plot_confusion_matrices(confusion_matrix, fitted_models, metrics, study, cmap)

print(f"Confusion matrices saved successfully for {study}")