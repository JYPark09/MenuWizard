import torch
from torch.utils.data import TensorDataset

def load_labels():
    labels = []

    with open('./data/label.csv', 'r') as f:
        for i, line in enumerate(f.readlines(), 0):
            if i:
                labels.append(line.split(',')[1].strip())

    return labels

def load_data(filename):
    X, Y = [], []

    with open(filename, 'r') as f:
        for line in f.readlines():
            line = list(map(float, line.strip().split(',')))

            X.append(line[1:])
            Y.append(line[0])

    X = torch.tensor(X).float().view(-1, 38)
    Y = torch.tensor(Y).long()

    return TensorDataset(X, Y)
