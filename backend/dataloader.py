import torch
from torch.utils.data import TensorDataset

import numpy as np

def normalize(arr, temp_mean, temp_var, time_mean, time_var):
    arr[2] = (arr[2] - temp_mean) / temp_var
    arr[4] = (arr[4] - time_mean) / time_var

def load_labels():
    labels = []

    with open('./data/label.csv', 'r') as f:
        for i, line in enumerate(f.readlines(), 0):
            labels.append(line.split(',')[1].strip())

    return labels

def load_data(filename):
    X, Y = [], []

    with open(filename, 'r') as f:
        for line in f.readlines():
            line = list(map(float, line.strip().split(',')))

            X.append(line[1:])
            Y.append(line[0])

    means = np.mean(X, axis=0)
    vars = np.var(X, axis=0)

    temp_mean = means[2]
    temp_var = vars[2]

    time_mean = means[4]
    time_var = vars[4]

    for i in range(len(X)):
        normalize(X[i], temp_mean, temp_var, time_mean, time_var)

    X = torch.tensor(X).float().view(-1, 38)
    Y = torch.tensor(Y).long()

    return TensorDataset(X, Y), temp_mean, temp_var, time_mean, time_var
