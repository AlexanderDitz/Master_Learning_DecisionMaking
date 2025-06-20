import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from benchmarking.hierarchical_bayes_numpyro import main
from benchmarking.hierarchical_bayes_dezfouli2019 import main

# import jax
# jax.config.update('jax_disable_jit', True)

# models = ['ApBrBch']
# data = 'data/data_128p_0.csv'
# output_file = 'params/hbi_test.nc'
# train_test_ratio = 1.0

# models = ['ApBr', 'ApBrAcfpBcf', 'ApBrAcfpBcfBch', 'ApAnBrBch', 'ApAnBrAcfpAcfnBcfBch', 'ApAnBrBcfBch']
# data = 'data/eckstein2022/eckstein2022.csv'
# output_file = 'params/eckstein2022_long_lr_0_001/mcmc_eckstein2022.nc'
# train_test_ratio = 0.8

models = ['ApAnBrBcfAchBch']
data = 'data/dezfouli2019/dezfouli2019.csv'
output_file = 'params/dezfouli2019/mcmc_dezfouli2019.nc'
train_test_ratio = [3, 6, 9]

num_samples = 5000
num_warmup = 1000
num_chains = 2
hierarchical = True

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