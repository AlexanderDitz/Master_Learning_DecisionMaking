import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

# models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']

models = ['ApBr', 'ApAnBcBr']

# model = 'ApBr'
num_samples = 4096
num_warmup = 1024
num_chains = 2
hierarchical = True
train_test_ratio = 0.5

data = 'data/eckstein2022/eckstein2022.csv'
output_file = 'params/eckstein2022/params_eckstein2022_baseline.nc'

# data = 'data/sugawara2021/sugawara2021.csv'
# output_file = 'params/sugawara2021/params_sugawara2021.nc'

data = 'data/parameter_recovery/data_128p_0.csv'
output_file = 'params/parameter_recovery/params_128p_0.nc'

for model in models:
    main(
        file=data,
        model=model,
        num_samples=num_samples,
        num_warmup=num_warmup, 
        num_chains=num_chains,
        hierarchical=hierarchical,
        output_file=output_file,
        train_test_ratio=train_test_ratio,
        checkpoint=False,
        )