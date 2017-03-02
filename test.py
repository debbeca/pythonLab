import pandas as pd
import numpy as np
import matplotlib
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

df.head()
df.describe()
X = f.ix[:,0:5];


le = LabelEncoder()

Y = f.ix[:,5];
Y_encoded = le.fit_transform(Y)
y_train = le.fit_transform(X['CI'][:-1])

model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(X_encoded, y_train)
