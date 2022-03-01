
# for each point
# find the k nearest neighbors
# assign the point to the largest category of neighbors

import numpy as np
import csv
from matplotlib import pyplot as plt

data=None
header=None
point=None # which point needs prediction

def euclidean(a, b):
    A=np.asarray(a)
    B=np.asarray(b)
    A=np.delete(A, 0) # removing price
    B=np.delete(B, 0) # removing price
    return np.sqrt(np.sum(np.square(A-B)))

def manhattan(a, b):
    A=np.asarray(a)
    B=np.asarray(b)

    A=np.delete(A, 0) # removing price
    B=np.delete(B, 0) # removing price
    return np.sum(np.abs(A-B))

def compare(i):
    global point
    #print(point)
    return euclidean(i, point)

price_prediction=[]
def knn(k):
    global data
    data=data.tolist() # temporarily converting to list for custom comparator
    data.sort(key=compare)
    data=np.asarray(data)
    s=0
    for i in range(k):
        s += data[i][0]
    s/=k
    price_prediction.append(s)
   # print("price = ", s)
    
def normalize():
    for i in range(1, len(data[0])):
        mn=1000000000000
        mx=0
        for j in range(len(data)):
            mn=min(mn, data[j][i])
            mx=max(mx, data[j][i])
        for j in range(len(data)):
            data[j][i]=(data[j][i]-mn)/(mx-mn)

def setup():
    global data, header
    with open("data/new.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            line_data=[i for line, i in enumerate(row)]
            data=(line_data if data is None else np.vstack([data, line_data]))
           # print(data)
    
    # delete the names
    data=data[:,1:]
    # delete header
    header=data[0]
    data=np.delete(data, (0), axis=0)
    # convert strings to floats
    data=data.astype(np.float)
    print(data)

setup()
point=np.array([0, 3145,236,528,0.17,9,7,6])

#normalize()

for k in range(1, len(data)-1):
    knn(k)

plt.plot(price_prediction)
plt.show()

#knn(1, )

#print(header)
#print(data)

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier

print("hello1")

"""