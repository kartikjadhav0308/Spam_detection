import pandas as pd
import numpy as np

df = pd.read_csv(r"DataSet\spam.csv",encoding = "latin1")
# print(df.head(10))
# print(df.info())

#drop the last column which having the more null values
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
# print(df.info())

