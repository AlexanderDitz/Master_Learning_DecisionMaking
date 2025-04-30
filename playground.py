import pandas as pd

df = pd.read_csv('all_scores.csv', index_col=0)
print(df['RNN'].mean())