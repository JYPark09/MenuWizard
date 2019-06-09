import torch
from torch import nn
import torch.nn.functional as F

INPUT_SIZE = 38
OUTPUT_SIZE = 32

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, OUTPUT_SIZE)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)

        return F.log_softmax(self.fc3(x), dim=1)
