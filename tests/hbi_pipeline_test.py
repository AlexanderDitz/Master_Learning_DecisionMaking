import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

# models = ['ApBr', 'ApBrAcfpBcf', 'ApBrAcfpBcfBch', 'ApAnBrBch', 'ApAnBrAcfpAcfnBcfBch', 'ApAnBrBcfBch']
models = ['ApBrAcfpBcfBch']
data = 'data/eckstein2022/eckstein2022.csv'
output_file = 'params/eckstein2022/mcmc_eckstein2022.nc'

num_samples = 5000
num_warmup = 1000
num_chains = 2
hierarchical = True
train_test_ratio = 1.0

for model in models:
    print("Fitting model", model, "...")
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