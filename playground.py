import numpy as np

# Example choices array
choices = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 0])

# Find switch points
switches = np.where(np.diff(choices) != 0)[0]

# Create boolean index array
index_array = np.zeros_like(choices, dtype=bool)

# Mark switch points and next three trials
for switch in switches:
    index_array[switch:switch+2] = True  # Mark switch and next three trials

# Ensure we don't go out of bounds
index_array = index_array[:len(choices)]

print(index_array)