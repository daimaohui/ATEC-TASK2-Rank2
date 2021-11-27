import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(165, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x=self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(166, 128)
        self.layer2 = nn.Linear(128, 40)
        self.layer3 = nn.Linear(40, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x