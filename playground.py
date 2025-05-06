import pandas as pd
import numpy as np

df = pd.read_csv('all_scores.csv', index_col=0)

n_trials = df['Trials'].values
spice_nll = df['SPICE'].values

print((spice_nll/n_trials).mean())

n_samples = 100
nlls = []
for _ in range(10):
    # get n_samples random samples
    index_samples = np.random.randint(0, len(df), n_samples)
    nlls.append((spice_nll[index_samples]/n_trials[index_samples]).mean())

print(nlls)
print(np.mean(nlls))
print(np.std(nlls))