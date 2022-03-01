import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('new_cut.csv')
data = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'explicit', 'year']]
print(data.head(5))

X = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
y = data[['explicit', 'year']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = StandardScaler()
scaler.fit(X_train)

StandardScaler()

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.tolist()[0:2])
print(X_test.tolist()[0:2])