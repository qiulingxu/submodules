'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from cl import ClassificationMask as CM
from cl.utils import get_config

class LeNet(nn.Module):
    def __init__(self, procfunc=None, num_classes=10):
        super(LeNet, self).__init__()
        self.procfunc = procfunc
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        if self.procfunc is not None:
            x = self.procfunc(x)        
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


CMLeNet = CM(LeNet)

def CLeNet(**karg):
    print(get_config("CLASS_NUM"))
    return CMLeNet(num_classes=get_config("CLASS_NUM"))

