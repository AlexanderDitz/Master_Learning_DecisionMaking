import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/martynaplomecka/closedloop_rl/data/eckstein2022/SLCNinfo_Share.csv'
df = pd.read_csv(file_path)

print("=== SLCNinfo_Share.csv DATASET ===")
print(f"Total subjects: {len(df)}")
print(f"All subject IDs: {sorted(df['ID'].tolist())}")
print()

# Apply conditions and show IDs at each step
print("=== APPLYING CATEGORIZATION CONDITIONS ===")

# Condition 1: ID 1-221
condition1_mask = (df['ID'] >= 1) & (df['ID'] <= 221)
condition1_ids = df[condition1_mask]['ID'].tolist()
print(f"Condition 1 (ID 1-221): {len(condition1_ids)} subjects - {sorted(condition1_ids)}")
print()

# Condition 2: ID 300-365
condition2_mask = (df['ID'] >= 300) & (df['ID'] <= 365)
condition2_ids = df[condition2_mask]['ID'].tolist()
print(f"Condition 2 (ID 300-365): {len(condition2_ids)} subjects - {sorted(condition2_ids)}")
print()

# Condition 3: ID > 400 and age < 23
condition3_mask = (df['ID'] > 400) & (df['age - years'] < 23)
condition3_ids = df[condition3_mask]['ID'].tolist()
print(f"Condition 3 (ID > 400 and age < 23): {len(condition3_ids)} subjects - {sorted(condition3_ids)}")
print()

# Apply categorization
conditions = [condition1_mask, condition2_mask, condition3_mask]
choices = [1, 2, 3]
df['Category'] = np.select(conditions, choices, default=0)

print("=== AFTER CATEGORIZATION ===")
all_categorized_ids = df['ID'].tolist()
print(f"All subjects with categories: {len(all_categorized_ids)} subjects - {sorted(all_categorized_ids)}")
print()

# Swap categories 2 and 3
print("=== SWAPPING CATEGORIES 2 AND 3 ===")
df['Category'] = df['Category'].replace({2: 3, 3: 2})

print("After category swap:")
all_swapped_ids = df['ID'].tolist()
print(f"All subjects after swap: {len(all_swapped_ids)} subjects - {sorted(all_swapped_ids)}")
print()

print(df['Category'].value_counts())

# Filter out uncategorized subjects
print("=== FILTERING OUT UNCATEGORIZED SUBJECTS ===")
uncategorized_mask = df['Category'] == 0
uncategorized_ids = df[uncategorized_mask]['ID'].tolist()
print(f"Uncategorized subjects (Category 0): {len(uncategorized_ids)} subjects - {sorted(uncategorized_ids)}")
print()

df_filtered = df[df['Category'] != 0].copy()
print(f"SLCNinfo_Share.csv dataset: {len(df)} subjects")
print(f"After removing uncategorized: {len(df_filtered)} subjects")
print(f"Removed: {len(df) - len(df_filtered)} subjects")
print()

print("=== SLCN.csv FILTERED DATASET ===")
print(f"All remaining subject IDs: {len(df_filtered)} subjects - {sorted(df_filtered['ID'].tolist())}")
print()

# Save filtered dataset
output_path = '/Users/martynaplomecka/closedloop_rl/data/eckstein2022/SLCN.csv'
df_filtered.to_csv(output_path, index=False)
print(f"Filtered dataset saved: {output_path}")

# Create visualization
colors = {1: 'red', 2: 'blue', 3: 'green'}  
plt.figure(figsize=(12, 8))

for category in [1, 2, 3]:
    mask = df_filtered['Category'] == category
    if mask.any():
        plt.scatter(
            df_filtered[mask]['ID'],
            df_filtered[mask]['age - years'],
            c=colors[category],
            label=f'Category {category}',
            alpha=0.7,
            s=30
        )

plt.xlabel('ID')
plt.ylabel('Age (years)')
plt.title('Subject ID vs Age Category')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_plots'
output_file = os.path.join(output_dir, 'id_vs_age_plot.png')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nFiltered dataset - ID range: {df_filtered['ID'].min()} - {df_filtered['ID'].max()}")
print(f"Filtered dataset - Age range: {df_filtered['age - years'].min()} - {df_filtered['age - years'].max()} years")

print("\n=== SLCN.csv STATISTICS ===")
for category in [1, 2, 3]:
    category_data = df_filtered[df_filtered['Category'] == category]
    category_ids = category_data['ID'].tolist()
    print(f"\nCategory {category}: {len(category_data)} subjects - {sorted(category_ids)}")
    if len(category_data) > 0:
        print(f"ID range: {category_data['ID'].min()} - {category_data['ID'].max()}")
        print(f"Age range: {category_data['age - years'].min()} - {category_data['age - years'].max()} years")