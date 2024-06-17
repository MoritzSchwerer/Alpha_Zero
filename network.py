import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Tuple


def conv2d(in_channels: int, out_channels: int, bias: bool = False):
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1,1), bias=bias, padding=(1,1)
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, interm_channels: int, use_bn: bool = True):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.interm_channels = interm_channels
        self.use_bn = use_bn

        self.net = nn.Sequential(
            *[
                conv2d(in_channels, self.interm_channels, bias=not self.use_bn),
                nn.BatchNorm2d(self.interm_channels) if self.use_bn else nn.Identity(),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, use_bn: bool = True):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.use_bn = use_bn

        self.conv1 = conv2d(in_channels, in_channels, bias=not self.use_bn)
        self.bn1 = nn.BatchNorm2d(in_channels) if self.use_bn else nn.Identity()
        self.conv2 = conv2d(in_channels, in_channels, bias=not self.use_bn)
        self.bn2 = nn.BatchNorm2d(in_channels) if self.use_bn else nn.Identity()

    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + inp)


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        total_num_actions: int = 4672,
        use_bn: bool = True,
        num_blocks: int = 19,
        interm_channels: int = 256,
        observation_space: Tuple[int, ...] = (111, 8, 8),  # chess
    ):
        super(PredictionNetwork, self).__init__()
        self.in_channels = in_channels
        self.interm_channels = interm_channels
        self.total_num_actions = total_num_actions
        self.use_bn = use_bn
        self.num_blocks = num_blocks
        self.observation_space = observation_space

        self.shared_net = nn.Sequential(
            ConvBlock(in_channels, interm_channels, use_bn),
            *[ResBlock(interm_channels, use_bn) for _ in range(num_blocks)],
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.interm_channels,
                out_channels=self.interm_channels,
                kernel_size=1,
                bias=not self.use_bn,
            ),
            nn.BatchNorm2d(self.interm_channels) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                math.prod(self.observation_space[-2:]) * self.interm_channels,
                total_num_actions,
            ),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.interm_channels,
                out_channels=self.interm_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.interm_channels) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * self.interm_channels, self.interm_channels),
            nn.ReLU(),
            nn.Linear(self.interm_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        feats = self.shared_net(x)
        policy = self.policy_head(feats)
        value = self.value_head(feats)
        return policy, value
