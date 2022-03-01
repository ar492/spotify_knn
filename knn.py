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


data = pd.read_csv('new cut.csv')
