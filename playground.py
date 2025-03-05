import numpy as np
import matplotlib.pyplot as plt
import ot  # Optimal transport library

# Simulate observed reaction times (RTs)
np.random.seed(42)
reaction_times = np.random.lognormal(mean=2.5, sigma=0.5, size=500)

# Define initial distribution: assume uniform in this case
num_bins = 500  # Set the number of bins to ensure matching sizes
initial_distribution = np.linspace(0, 10, num_bins)  # Uniformly spaced support
initial_probs = np.ones_like(initial_distribution) / len(initial_distribution)

# Define target distribution: reaction times histogram with matching bins
bins = np.linspace(0, 10, num_bins)
target_probs, _ = np.histogram(reaction_times, bins=bins, density=True)
target_distribution = initial_distribution  # Ensure matching support

# Normalize probabilities
target_probs /= np.sum(target_probs)

# Compute the cost matrix for Brownian motion
cost_matrix = ot.utils.dist(initial_distribution[:, None], target_distribution[:, None]) ** 2

# Solve for entropy-regularized transport plan using Sinkhorn iterations
reg = 1e-3  # Regularization parameter
transport_plan = ot.sinkhorn(initial_probs, target_probs, cost_matrix, reg)

# Interpolate distributions using barycenter
# Create a stacked array for the initial and target distributions
A = np.vstack([initial_probs, target_probs]).T

steps = 10  # Number of interpolation steps
weights_list = np.linspace(0, 1, steps + 1)  # Interpolation weights
interpolations = [
    ot.barycenter(A, cost_matrix, reg, weights=np.array([1 - t, t]))
    for t in weights_list
]

# Plot results
plt.figure(figsize=(12, 6))

# Plot initial, target, and interpolated distributions
plt.plot(initial_distribution, initial_probs, label="Initial Distribution", linestyle="--")
plt.plot(target_distribution, target_probs, label="Target Distribution (Reaction Times)")
for i, interp in enumerate(interpolations):
    plt.plot(initial_distribution, interp, alpha=0.4, label=f"Step {i}" if i in [0, steps] else None)

plt.xlabel("State (Reaction Time)")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Transition from Initial to Target Distribution using Barycenter Interpolation")
plt.show()
