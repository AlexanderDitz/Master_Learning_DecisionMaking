import numpy as np
from utils.convert_dataset import convert_dataset

file = 'data/eckstein2022/eckstein2022_age.csv'

dataset, _, df, _ = convert_dataset(file)

print(dataset.xs[:, 0, -1].unique())
print(df['session'].unique())