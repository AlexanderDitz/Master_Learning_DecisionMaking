import numpy as np

n_trials = 1000

n_trials_row = 20
zeros = np.zeros(n_trials_row)
ones = np.ones(n_trials_row)
trials = []

for i in range(n_trials//n_trials_row):
    if i%2==0:
        trials.append(ones)
    else:
        trials.append(zeros)
trials = np.concatenate(trials)
print(trials[0], trials[-1])