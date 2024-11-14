import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr=0.001):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Linear(8, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.2)

        self._create_weights()


    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        return x