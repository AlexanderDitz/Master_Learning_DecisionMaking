import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main
from resources.model_evaluation import plot_traces

models = ['ApBr','ApAnBr','ApBrBc','ApBrAcBc', 'ApAnBrBc', 'ApAnBrAcBc']

data = 'data/sugawara2021/sugawara2021_sindy.csv'
output_file = f'params/sugawara2021/params_sugawara2021_mcmc.nc'

# data = 'data/2arm/eckstein2022_291_processed.csv'
# output_file = f'benchmarking/params/eckstein2022_291/traces.nc'

# data = 'data/2arm/data_rnn_br30_a025_ap05_bch30_ach05_varDict.csv'
# output_file = f'benchmarking/params/traces_test.nc'

for model in models:
    mcmc = main(
        file=data, 
        model=model, 
        num_samples=4096, 
        num_warmup=1024, 
        num_chains=2,
        hierarchical=True,
        output_file=output_file.replace('.', f'_{model}.'),
        checkpoint=False,
        )
    
    # mcmc.print_summary()
    # plot_traces(mcmc)