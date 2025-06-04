import numpy as np
from utils.convert_dataset import convert_dataset

file = 'data/eckstein2022/eckstein2022_age_gender.csv'

dataset = convert_dataset(file)[0]

print(len(dataset.xs[:, 0, -1].unique()))