import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

models = ['ApBr', 'ApAnBcBr']
num_samples = 4096
num_warmup = 1024
num_chains = 2
hierarchical = True

data = 'data/parameter_recovery/data_128p_0.csv'
output_file = 'params/parameter_recovery/params_128p_0.nc'

for simulated_model in models:
    for fitted_model in models:
    
        print(f'Fitting {fitted_model} model to {simulated_model} data...')

        main(
            file=data.replace('.', f'_{simulated_model}.'), 
            model=fitted_model, 
            num_samples=num_samples, 
            num_warmup=num_warmup, 
            num_chains=num_chains,
            hierarchical=hierarchical,
            output_file=output_file.replace('.', f'_{simulated_model}.'),
            checkpoint=False,
            )