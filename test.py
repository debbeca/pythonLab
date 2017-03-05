import pandas as pd
import numpy as np
import matplotlib
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

df.head()
df.describe()
var_mod = ['CI','D','PR']
le = LabelEncoder()
df['CI'] = le.fit_transform(df['CI'])
df['D'] = le.fit_transform(df['D'])
df['PR']= le.fit_transform(df['PR'])

