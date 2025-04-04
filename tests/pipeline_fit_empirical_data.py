import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main as mcmc_training
from utils.create_dataset_AgentMCMC import main as create_dataset_mcmc
from utils.create_dataset_AgentSPICE import main as create_dataset_spice
from pipeline_rnn import main as rnn_training

# MCMC training configuration
num_samples = 4096
num_warmup = 1024
num_chains = 2
hierarchical = True
training_test_ratio = 0.8

# SPICE training configuration
epochs = 4096
scheduler = True

# Data generation configuration
n_trials_per_session = 500

# empirical data
data = 'data/eckstein2022/eckstein2022.csv'

# ----------------------------------------------------------------------------------
# Fitting SPICE to empricial data
# ----------------------------------------------------------------------------------

rnn_training(
    checkpoint=False,
    epochs=epochs,
    
    model = f"params/eckstein2022/params_eckstein2022.pkl",
    data = data,
    
    n_actions=2,
    
    # optuna params for GRU
    embedding_size=22,
    n_steps=32,
    learning_rate=0.00023,
    
    batch_size=-1,
    sequence_length=-1,
    train_test_ratio=1.0,
    scheduler=scheduler,
    bagging=True,
    
    n_sessions=128,
    n_trials=200,
    sigma=0.2,
    analysis=False,
    )


# ----------------------------------------------------------------------------------
# Fitting MCMC models to empricial data
# ----------------------------------------------------------------------------------

# models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']
models = ['ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr']
output_file = 'params/eckstein2022/params_eckstein2022.nc'

for model in models:
    print(f"Fitting model {model} to empirical data...")

    mcmc_training(
        file=data, 
        model=model, 
        num_samples=num_samples, 
        num_warmup=num_warmup, 
        num_chains=num_chains,
        hierarchical=hierarchical,
        output_file=output_file,
        train_test_ratio=training_test_ratio,
        checkpoint=False,
        )


# ----------------------------------------------------------------------------------
# Simulating data with fitted model
# ----------------------------------------------------------------------------------

models = ['ApBr', 'ApAnAcBcBr', 'Spice']

for model in models:
    print(f"Generating data with model {model}...")
    if model != 'Spice':
        create_dataset_mcmc(
            path_model='params/eckstein2022/params_eckstein2022_'+model+'.nc',
            path_data=data,
            path_save=data.replace('.', '_training_'+model+'.'),
            n_trials_per_session=n_trials_per_session,
            )
        create_dataset_mcmc(
            path_model='params/eckstein2022/params_eckstein2022_'+model+'.nc',
            path_data=data,
            path_save=data.replace('.', '_testing_'+model+'.'),
            n_trials_per_session=n_trials_per_session,
            )
    else:
        create_dataset_spice(
            path_model='params/eckstein2022/params_eckstein2022.pkl',
            path_data=data,
            path_save=data.replace('.', '_training_'+model+'.'),
            n_trials_per_session=n_trials_per_session,
        )
        create_dataset_spice(
            path_model='params/eckstein2022/params_eckstein2022.pkl',
            path_data=data,
            path_save=data.replace('.', '_testing_'+model+'.'),
            n_trials_per_session=n_trials_per_session,
        )

# ----------------------------------------------------------------------------------
# Fitting models to simulated data
# ----------------------------------------------------------------------------------

for simulated_model in models:
    data = f"data/eckstein2022/eckstein2022_training_{simulated_model}.csv"
    for fitted_model in models:
        if fitted_model != 'Spice':
            print(f"Fitting model {fitted_model} to data {simulated_model}...")
            output_file = f"params/eckstein2022/params_eckstein2022_{simulated_model}.nc"

            mcmc_training(
                file=data, 
                model=fitted_model, 
                num_samples=num_samples, 
                num_warmup=num_warmup, 
                num_chains=num_chains,
                hierarchical=hierarchical,
                output_file=output_file,
                checkpoint=False,
                )
        else:
            rnn_training(
                checkpoint=False,
                epochs=epochs,
                
                model = f"params/eckstein2022/params_eckstein2022_{simulated_model}_{fitted_model}.pkl",
                data = data,
                
                n_actions=2,
                
                # optuna params for GRU
                embedding_size=22,
                n_steps=32,
                learning_rate=0.00023,
                
                batch_size=-1,
                sequence_length=-1,
                train_test_ratio=1.0,
                scheduler=scheduler,
                bagging=True,
                
                n_sessions=128,
                n_trials=200,
                sigma=0.2,
                analysis=False,
            )