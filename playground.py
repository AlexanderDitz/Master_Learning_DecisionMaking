import itertools

def group_sequences_by_target(target_sequence, *sequences):
    # Use itertools.groupby to find the indices of groups in the target sequence
    groups = [(key, list(group)) for key, group in itertools.groupby(enumerate(target_sequence), key=lambda x: x[1])]
    
    # Extract the indices for each group
    group_indices = [list(map(lambda x: x[0], group)) for _, group in groups]
    
    # Group each input sequence based on the indices
    grouped_sequences = []
    for seq in sequences:
        grouped_sequences.append([[seq[i] for i in indices] for indices in group_indices])
    
    return grouped_sequences

# Example usage
binary_target = [1, 1, 0, 0, 1, 1, 1, 0]
sequence1 = [10, 20, 30, 40, 50, 60, 70, 80]
sequence2 = [5, 15, 25, 35, 45, 55, 65, 75]

grouped_sequences = group_sequences_by_target(binary_target, sequence1, sequence2)

# Output the grouped sequences
for idx, seq in enumerate(grouped_sequences, start=1):
    print(f"Grouped Sequence {idx}: {seq}")