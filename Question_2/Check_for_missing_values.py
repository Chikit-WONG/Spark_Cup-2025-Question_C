import pandas as pd

df = pd.read_csv("./Data/input_data/df_movies_train.csv")
print(df.isnull().sum())
print()
print(df.isna().sum())  # 两者等价，输出每列的缺失值数量
