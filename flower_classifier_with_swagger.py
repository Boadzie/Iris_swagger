# -*- coding: utf-8 -*-
"""Flower Classifier with Swagger.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sa1ieTKeVSuSbfqbPc3SwhMHW7wx0-MJ
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3 )

model = RandomForestClassifier()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

# Accuracy
print(accuracy_score(predicted, y_test))

import pickle

with open('model.pkl', 'wb') as model_pickle:
  pickle.dump(model, model_pickle)

