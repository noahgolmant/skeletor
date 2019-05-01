'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3,
                 width=32, height=32):
        super(LeNet, self).__init__()
        width, height = width - 4, height - 4
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        width, height = width // 2, height // 2  # max pool
        self.conv2 = nn.Conv2d(6, 16, 5)
        width, height = width - 4, height - 4
        width, height = width // 2, height //2  # max pool again
        in_dim = 16 * width * height

        self.fc1   = nn.Linear(in_dim, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
