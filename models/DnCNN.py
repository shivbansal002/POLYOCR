import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64, depth=20):
        super(DnCNN, self).__init__()

        self.in_conv = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv_list = nn.ModuleList()
        for _ in range(depth - 2):
            self.conv_list.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.in_conv(x))
        for conv in self.conv_list:
            out = self.relu(conv(out))
        out = self.out_conv(out)
        return out

    


