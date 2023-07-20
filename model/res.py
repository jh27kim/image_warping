import torch.nn as nn
import torch.nn.functional as F

class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=4,
                               kernel_size=3,
                               stride=1,
                               padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = out + x

        return out