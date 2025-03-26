import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example models: A, B, and C (defined as functions to simulate and fit data)
def model_A_simulate(params, n_trials=100):
    # Simulate binary choice data based on model A
    return np.random.choice([0, 1], size=n_trials, p=params)

def model_B_simulate(params, n_trials=100):
    # Simulate binary choice data based on model B
    return np.random.choice([0, 1], size=n_trials, p=params)

def model_C_simulate(params, n_trials=100):
    # Simulate binary choice data based on model C
    return np.random.choice([0, 1], size=n_trials, p=params)

def fit_model(data):
    # Fit a model to the data and return the best-fitting model (A, B, or C)
    likelihoods = {
        "A": np.random.rand(),  # Placeholder for likelihood computation
        "B": np.random.rand(),
        "C": np.random.rand()
    }
    return max(likelihoods, key=likelihoods.get)

# Parameters for each model
params_A = [0.7, 0.3]  # Probabilities for model A
params_B = [0.5, 0.5]  # Probabilities for model B
params_C = [0.2, 0.8]  # Probabilities for model C

# Simulation settings
n_simulations = 100
n_trials = 100
models = ["A", "B", "C"]
simulate_functions = {
    "A": lambda: model_A_simulate(params_A, n_trials),
    "B": lambda: model_B_simulate(params_B, n_trials),
    "C": lambda: model_C_simulate(params_C, n_trials),
}

# Simulate data and fit models
true_labels = []
fit_labels = []
for sim_model in models:
    for _ in range(n_simulations):
        data = simulate_functions[sim_model]()
        fit_model_label = fit_model(data)
        true_labels.append(sim_model)
        fit_labels.append(fit_model_label)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, fit_labels, labels=models, normalize="true")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, xticklabels=models, yticklabels=models, cmap="Blues")
plt.xlabel("Fitted Model")
plt.ylabel("Simulated Model")
plt.title("Confusion Matrix: p(Fitted Model | Simulated Model)")
plt.show()
