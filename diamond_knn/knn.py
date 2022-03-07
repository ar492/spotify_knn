import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def normalize(df):
	result = df.copy()
	for feature_name in df.columns:
		max_value=df[feature_name].max()
		min_value=df[feature_name].min()
		result[feature_name]=(df[feature_name]-min_value)/(max_value-min_value)
	return result

data = pd.read_csv('pruned.csv')
data=normalize(data)

X = data[['size', 'depth_percent', 'table_percent', 'meas_length', 'meas_width', 'meas_depth']]
y = data[['total_sales_price']]

print(X.head(5))
print(y.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn=KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
# knn.fit(X_train, np.ravel(y_train, order='C'))
y_pred = knn.predict(X_test)
# print(accuracy_score(y_test, y_pred))




"""
print(" test")
print(y_test)
print("pred")
print(y_pred)
"""


xs=[i for i in range(len(y_test))]
assert(len(y_test)==len(y_pred))


"""
difference = np.subtract(y_test, y_pred)
squared = np.square(difference)
mse = squared.mean()
print(mse)
print("accuracy: ", sum(1 for x,y in zip(y_test,y_pred) if x == y) / len(y_pred))
"""

from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error(y_test, y_pred))

print(accuracy_score(y_test, y_pred))


