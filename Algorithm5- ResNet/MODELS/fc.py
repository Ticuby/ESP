import torch.nn as nn
import torch

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc1 = nn.Linear(3*224*224,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,2)
    def forward(self,x):
        x = torch.flatten(x, 1)
        x = nn.ReLU()(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x
