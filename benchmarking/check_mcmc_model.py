import pickle
import arviz as az
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import benchmarking_dezfouli2019, benchmarking_eckstein2022

# path_model = 'params/eckstein2022/mcmc_eckstein2022_ApBrAcfpBcfBch.nc' 
# rl_model = benchmarking_eckstein2022.rl_model'

path_model = 'params/dezfouli2019/mcmc_dezfouli2019_baseline.nc'
gql_model = benchmarking_dezfouli2019.gql_model
params = ['phi', 'beta']

# setup mcmc agent
with open(path_model, 'rb') as file:
    mcmc = pickle.load(file)

posterior_samples = mcmc.get_samples()
idata = az.from_numpyro(mcmc, log_likelihood=False)

summary = az.summary(idata, var_names=None)
print(summary['r_hat'])

# get mean values of each parameter
param_container = {p: None for p in params}
for param in params:
    param_container[param] = mcmc.get_samples()[param].mean(axis=0).mean(axis=-1)
print(param_container)

# Save to a CSV file
summary.to_csv("summary.csv")

az.plot_trace(idata)
plt.show()
