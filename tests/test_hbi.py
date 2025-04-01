import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

# models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']

model = 'ApAnBcBr'
num_samples = 4096
num_warmup = 1024
num_chains = 2
hierarchical = True
train_test_ratio = 0.8

data = 'data/eckstein2022/eckstein2022.csv'
output_file = f'params/eckstein2022/params_eckstein2022.nc'

# data = 'data/sugawara2021/sugawara2021.csv'
# output_file = f'params/sugawara2021/params_sugawara2021.nc'

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