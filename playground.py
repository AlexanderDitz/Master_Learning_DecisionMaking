import pandas as pd
import numpy as np

file = "data/raw_data/sugawara2021.csv"

df = pd.read_csv(file)

print(np.unique(df[df['type']==4]['choice'].values))