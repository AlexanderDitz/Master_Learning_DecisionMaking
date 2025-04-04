import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

model = 'ApAnBrAcBc'
num_samples = 4096
num_warmup = 512
num_chains = 2
hierarchical = True

print('Fitting MCMC model to baseline data...')
data = 'data/sugawara2021/sugawara2021_ApBr.csv'
output_file = f'params/sugawara2021/params_sugawara2021_ApBr_'+model+'.nc'

main(
    file=data, 
    model=model, 
    num_samples=num_samples, 
    num_warmup=num_warmup, 
    num_chains=num_chains,
    hierarchical=hierarchical,
    output_file=output_file,
    checkpoint=False,
    )

print('Fitting MCMC model to MCMC data...')
data = 'data/sugawara2021/sugawara2021_'+model+'.csv'
output_file = f'params/sugawara2021/params_sugawara2021_'+model+'_'+model+'.nc'

main(
    file=data, 
    model=model, 
    num_samples=num_samples, 
    num_warmup=num_warmup, 
    num_chains=num_chains,
    hierarchical=hierarchical,
    output_file=output_file,
    checkpoint=False,
    )

print('Fitting MCMC model to SPICE data...')
data = 'data/sugawara2021/sugawara2021_spice.csv'
output_file = f'params/sugawara2021/params_sugawara2021_spice_'+model+'.nc'

main(
    file=data, 
    model=model, 
    num_samples=num_samples, 
    num_warmup=num_warmup, 
    num_chains=num_chains,
    hierarchical=hierarchical,
    output_file=output_file,
    checkpoint=False,
    )

model = 'ApBr'
print('Fitting baseline model to baseline data...')
data = 'data/sugawara2021/sugawara2021_ApBr.csv'
output_file = f'params/sugawara2021/params_sugawara2021_ApBr_ApBr.nc'

main(
    file=data, 
    model=model, 
    num_samples=num_samples, 
    num_warmup=num_warmup, 
    num_chains=num_chains,
    hierarchical=hierarchical,
    output_file=output_file,
    checkpoint=False,
    )

print('Fitting baseline model to MCMC data...')
data = 'data/sugawara2021/sugawara2021_ApAnBrAcBc.csv'
output_file = f'params/sugawara2021/params_sugawara2021_ApAnBrAcBc_ApBr.nc'

main(
    file=data, 
    model=model, 
    num_samples=num_samples, 
    num_warmup=num_warmup, 
    num_chains=num_chains,
    hierarchical=hierarchical,
    output_file=output_file,
    checkpoint=False,
    )

print('Fitting baseline model to SPICE data...')
data = 'data/sugawara2021/sugawara2021_spice.csv'
output_file = f'params/sugawara2021/params_sugawara2021_spice_ApBr.nc'

main(
    file=data, 
    model=model, 
    num_samples=num_samples, 
    num_warmup=num_warmup, 
    num_chains=num_chains,
    hierarchical=hierarchical,
    output_file=output_file,
    checkpoint=False,
    )