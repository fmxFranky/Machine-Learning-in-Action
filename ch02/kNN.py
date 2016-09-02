import numpy as np
import operator
import pandas as pd
import sys


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def loadSet(file):
    return pd.read_table(file, header=None, names=['fly', ' game', 'ice', 'class'])


def normalize(dataSet):
    digitSection = dataSet[['fly', ' game', 'ice']]
    return digitSection.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


# TODO kNN Classifier:
print(normalize(loadSet('datingTestSet.txt')))