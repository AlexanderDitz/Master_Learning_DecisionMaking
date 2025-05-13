# import pandas as pd
# import numpy as np

# df = pd.read_csv('all_scores.csv', index_col=0)

# n_trials = df['Trials'].values
# spice_nll = df['RNN'].values

# # compute trial likelihood
# lik = np.exp(-spice_nll / (n_trials*2))

# print(lik)
# print(min(lik))

import numpy as np

milestone_pow_init = 9
milestones = np.cumsum([np.power(2, milestone_pow_init+i) for i in range(0, 6)])
print(milestones)