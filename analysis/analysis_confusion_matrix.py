import sys, os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_spice, setup_agent_mcmc, setup_agent_rnn
from resources.bandits import get_update_dynamics, AgentSpice, AgentNetwork
from resources.model_evaluation import log_likelihood, bayesian_information_criterion, akaike_information_criterion
from resources.sindy_utils import SindyConfig_eckstein2022 as SindyConfig
from resources.rnn import RLRNN_eckstein2022 as RLRNN
from utils.colormap import truncate_colormap


#----------------------------------------------------------------------------------------------
# CONFIGURATION CONFUSION MATRIX FILES
#----------------------------------------------------------------------------------------------

# dataset settings
dataset = "eckstein2022"
models = ["ApBr", "ApAnBrAcfpAcfnBcfBch"]

spice_model_name = "spice"

models.append(spice_model_name)

# file settings
path_data = f'data/{dataset}/{dataset}_test_SIMULATED.csv'
path_model_spice = f'params/{dataset}/confusion_matrix/rnn_{dataset}_SIMULATED.pkl'
path_model_mcmc = f'params/{dataset}/confusion_matrix/mcmc_{dataset}_SIMULATED_FITTED.nc'

n_actions = 2

#----------------------------------------------------------------------------------------------
# SETUP MODELS
#----------------------------------------------------------------------------------------------

simulated_models = deepcopy(models)
if 'spice' in simulated_models:
    simulated_models.remove('spice')
    simulated_models.append('rnn')

print("Simulated models:", simulated_models)
print("Fitted models:", models)

agents = {}
for simulated_model in simulated_models:
    agents[simulated_model] = {}
    for fitted_model in models:
        agents[simulated_model][fitted_model] = None
        
        if fitted_model.lower() != spice_model_name:
            agent = setup_agent_mcmc(
                path_model=path_model_mcmc.replace('SIMULATED', simulated_model).replace('FITTED', fitted_model)
                )
            n_params = 2 if fitted_model == "ApBr" else 6
        else:
            if spice_model_name == "rnn":
                agent = setup_agent_rnn(
                    class_rnn=RLRNN, 
                    path_model=path_model_spice.replace('SIMULATED', simulated_model), 
                    list_sindy_signals=SindyConfig['rnn_modules']+SindyConfig['control_parameters'],
                    )
                n_params = 12.647059  # avg n params of eckstein2022-SPICE models
            elif spice_model_name == "spice":
                agent = setup_agent_spice(
                    class_rnn=RLRNN,
                    path_spice=path_model_spice.replace('SIMULATED', simulated_model).replace('rnn', 'spice', 1),
                    path_rnn=path_model_spice.replace('SIMULATED', simulated_model),
                    path_data=path_data.replace('SIMULATED', fitted_model),
                    rnn_modules=SindyConfig['rnn_modules'],
                    control_parameters=SindyConfig['control_parameters'],
                    sindy_library_setup=SindyConfig['library_setup'],
                    sindy_filter_setup=SindyConfig['filter_setup'],
                    sindy_dataprocessing=SindyConfig['dataprocessing_setup'],
                    sindy_library_polynomial_degree=1,
                )
                n_params = 12.647059  # avg n params of eckstein2022-SPICE models
        agents[simulated_model][fitted_model] = (agent, n_params)


#----------------------------------------------------------------------------------------------
# PIPELINE CONFUSION MATRIX
#----------------------------------------------------------------------------------------------

metrics = ['nll', 'aic', 'bic']
confusion_matrix = {metric: np.zeros((len(models), len(models))) for metric in metrics}

for index_simulated_model, simulated_model in enumerate(simulated_models):
    
    # get data and choice probabilities from simulated model
    dataset = convert_dataset(file=path_data.replace('SIMULATED', simulated_model))[0]
    n_sessions = len(dataset)
    metrics_session = {metric: np.zeros((n_sessions, len(models))) for metric in metrics}

    for index_fitted_model, fitted_model in enumerate(models):
        
        print(f"Comparing fitted model {fitted_model} to simulated data from {simulated_model}...")
        
        # agent setup for fitted model
        agent, n_params = agents[simulated_model][fitted_model]
            
        for session in tqdm(range(n_sessions)):
            # get choice probabilities from agent for data from simulated model
            choice_probs_fitted = get_update_dynamics(experiment=dataset.xs[session], agent=agent if isinstance(agent, AgentSpice) or isinstance(agent, AgentNetwork) else agent[session])[1]
            choices = dataset.xs[session, :len(choice_probs_fitted), :n_actions].cpu().numpy()
            
            ll = log_likelihood(choices, choice_probs_fitted)
            
            metrics_session['nll'][session, index_fitted_model] = -ll
            metrics_session['bic'][session, index_fitted_model] = bayesian_information_criterion(choices, choice_probs_fitted, n_params, ll=ll)
            metrics_session['aic'][session, index_fitted_model] = akaike_information_criterion(choices, choice_probs_fitted, n_params, ll=ll)
    
    for metric in metrics:
        
        # get "best model"-counts for each model for each session
        best_model = np.argmin(metrics_session[metric], axis=-1)
        unique, counts = np.unique(best_model, return_counts=True)
        
        # Ensure all models have a count in the dictionary
        counts_dict = {model_index: 0 for model_index in range(len(models))}
        counts_dict.update(dict(zip(unique, counts / n_sessions)))
        
        confusion_matrix[metric][index_simulated_model] += np.array([counts_dict[key] for key in counts_dict])
    
# Plot confusion matrix

# Create figure and subplots
fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), constrained_layout=True)

cmap = truncate_colormap(plt.cm.viridis, minval=0.5, maxval=1.0)

# Plot NLL confusion matrix
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
        )
    axes[index_metric].set_xlabel("Fitted Model")
    axes[index_metric].set_ylabel("Simulated Model")
    axes[index_metric].set_title("Confusion Matrix: " + metric.upper())

# Show the figure
plt.savefig('analysis/plots_eckstein2022/confusion_matrix.png', dpi=500)
# plt.show()