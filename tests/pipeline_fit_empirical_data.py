import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main as mcmc_training
from utils.create_dataset_AgentMCMC import main as create_dataset_mcmc
from utils.create_dataset_AgentSPICE import main as create_dataset_spice
from pipeline_rnn import main as rnn_training
from pipeline_sindy import main as sindy_training


FIT_MCMC = 0
FIT_SPICE = 1
FIT_TO_EMPIRICAL = 0
FIT_TO_SIMULATED = 1
SIMULATE_DATA = 0

models = ['ApBr', 'ApAnBrAcfpAcfnBcfBch']

# MCMC training configuration
num_samples = 5000
num_warmup = 1000
num_chains = 2
hierarchical = True
training_test_ratio = 1.0

# SPICE training configuration
epochs = 65536
scheduler = True

# Data generation configuration
n_trials_per_session = 200

# empirical data
data = 'data/eckstein2022/eckstein2022.csv'

if FIT_TO_EMPIRICAL:
    
    if FIT_SPICE:
        # ----------------------------------------------------------------------------------
        # Fitting SPICE to empricial data
        # ----------------------------------------------------------------------------------
        
        rnn_training(
            checkpoint=False,
            epochs=epochs,
            
            model = f"params/eckstein2022/rnn_eckstein2022.pkl",
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

    if FIT_MCMC:
        # ----------------------------------------------------------------------------------
        # Fitting MCMC models to empricial data
        # ----------------------------------------------------------------------------------

        output_file = 'params/eckstein2022/mcmc_eckstein2022.nc'

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

if SIMULATE_DATA:
    # ----------------------------------------------------------------------------------
    # Simulating data with fitted model
    # ----------------------------------------------------------------------------------

    for model in models:
        print(f"Generating data with model {model}...")
        if model.lower() != 'spice':
            create_dataset_mcmc(
                path_model='params/eckstein2022/mcmc_eckstein2022_'+model+'.nc',
                path_data=data,
                path_save=data.replace('.', '_training_'+model+'.'),
                n_trials_per_session=n_trials_per_session,
                )
            create_dataset_mcmc(
                path_model='params/eckstein2022/mcmc_eckstein2022_'+model+'.nc',
                path_data=data,
                path_save=data.replace('.', '_testing_'+model+'.'),
                n_trials_per_session=n_trials_per_session,
                )
        else:
            create_dataset_spice(
                path_rnn='params/eckstein2022/spice_eckstein2022.pkl',
                path_data=data,
                path_save=data.replace('.', '_training_'+model+'.'),
                n_trials_per_session=n_trials_per_session,
            )
            create_dataset_spice(
                path_rnn='params/eckstein2022/spice_eckstein2022.pkl',
                path_data=data,
                path_save=data.replace('.', '_testing_'+model+'.'),
                n_trials_per_session=n_trials_per_session,
            )

if FIT_TO_SIMULATED:
    
    if FIT_MCMC:
    
        # ----------------------------------------------------------------------------------
        # Fitting MCMC models to simulated data
        # ----------------------------------------------------------------------------------

        for simulated_model in models+['spice']:
            data = f"data/eckstein2022/eckstein2022_training_{simulated_model}.csv"
            for fitted_model in models:
                print(f"Fitting MCMC model {fitted_model} to data of {simulated_model}...")
                output_file = f"params/eckstein2022/mcmc_eckstein2022_{simulated_model}.nc"
                mcmc_training(
                    file=data, 
                    model=fitted_model, 
                    num_samples=num_samples, 
                    num_warmup=num_warmup, 
                    num_chains=num_chains,
                    hierarchical=hierarchical,
                    output_file=output_file,
                    train_test_ratio=training_test_ratio,
                    checkpoint=False,
                    )
                
    if FIT_SPICE:

        # ----------------------------------------------------------------------------------
        # Fitting SPICE to simulated data
        # ----------------------------------------------------------------------------------

        for simulated_model in models+['spice']:
            data = f"data/eckstein2022/eckstein2022_training_{simulated_model}.csv"
            rnn_model = f"params/eckstein2022/rnn_eckstein2022_{simulated_model}.pkl"
            
            rnn_training(
                checkpoint=False,
                epochs=epochs,
                
                model = rnn_model,
                data = data,
                
                n_actions=2,
                
                embedding_size=32,
                n_steps=32,
                learning_rate=0.0001,
                l1_weight_decay=0.001,
                dropout=0.25,            
                batch_size=-1,
                sequence_length=-1,
                train_test_ratio=training_test_ratio,
                scheduler=scheduler,
                bagging=True,
                
                sigma=0.2,
                analysis=False,
            )
            
            sindy_training(
                model=rnn_model,
                data=data,
                save=True,
                use_optuna=True,
                pruning=True,
                train_test_ratio=training_test_ratio,
                polynomial_degree=1,
                optimizer_alpha=0.1,
                optimizer_threshold=0.05,
                n_trials_off_policy=1000,
                n_sessions_off_policy=1,
                n_trials_same_action_off_policy=5,
                verbose=False,
            )