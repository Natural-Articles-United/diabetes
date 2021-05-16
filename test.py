import pandas as pd

df = pd.read_csv('diabetes.csv')

test1 = df.iloc[:, 0:1].values

print(max(test1), min(test1))
