import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(40, 1, kernel_size=3, padding=1, stride=1)
        # self.pool = nn.MaxPool1d(2, stride=2)
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.pool(x)
        x = x.view(-1, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x